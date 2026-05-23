import threading
import time
import heapq
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Any, Tuple
import logging

logger = logging.getLogger(__name__)

@dataclass(order=True)
class QueueEvent:
    # Ordering is by t first, then by priority. action/metadata are not orderable.
    t: float
    priority: Tuple[int, ...]
    action: Callable[[], None] = field(compare=False)
    metadata: Any = field(default=None, compare=False)

class Scheduler(threading.Thread):
    def __init__(self, timing_policy: float = 0.98, daemon: bool = True):
        """
        :param timing_policy:
            0 -> Relative timing (wait the full scheduled delay),
            1 -> Absolute timing (cut wait time to catch up to the schedule),
            0.5 -> A blend of the two.
        """
        super().__init__(daemon=daemon)
        self.timing_policy = timing_policy

        # Heap-based queue for scheduled events.
        self._queue = []

        # Two synchronization primitives, kept deliberately separate. How each is used (and why) lives
        # where it matters — see run() for _queue_change_condition and while_quiescent() for _execution_lock.
        #   _queue_change_condition — guards the heap and signals changes to it: anything that reads or
        #       writes _queue holds it, and notify_all() wakes the run loop to re-evaluate the head.
        #   _execution_lock — held by the run loop for the full duration of each action's execution, so
        #       "held" == "mid-action". while_quiescent() takes it to mutate action-related state safely.
        # Lock order: the run loop never holds _queue_change_condition while taking _execution_lock — if
        # it did, a running action couldn't modify the queue. For the clock system that's essential:
        # actions routinely schedule wake-ups, fork, and reschedule after tempo changes.
        #
        # The condition uses a plain (non-reentrant) Lock on purpose: every critical section here is flat,
        # so accidental reentrancy would signal a bug and should deadlock loudly rather than be hidden by
        # Condition's default RLock. We don't need to keep a reference to the lock object, since
        # `with self._queue_change_condition:` acquires it directly.
        self._queue_change_condition = threading.Condition(threading.Lock())
        self._execution_lock = threading.Lock()
        self._killed = False

        # Timing variables.
        self._start_time = None
        self._last_wake_time = None
        self._ideal_time = 0.0  # Ideal scheduler time (seconds since start)

    def time(self, wake: bool = False) -> float:
        """Return the scheduler's ideal time (time that should have passed by schedule).

        If wake is True, the scheduler is notified to update its state.
        """
        if wake:
            with self._queue_change_condition:
                self._queue_change_condition.notify_all()
        return self._ideal_time

    def wall_time(self) -> float:
        """Return the actual wall-clock time elapsed since the scheduler started."""
        return time.time() - self._start_time if self._start_time else 0.0

    def kill(self) -> None:
        """Stop the scheduler."""
        self._killed = True
        # Wake the run loop wherever it's parked on the condition (empty-queue wait or timed wait)
        # so it observes _killed and exits.
        with self._queue_change_condition:
            self._queue_change_condition.notify_all()

    @contextmanager
    def while_quiescent(self):
        """
        Context manager that blocks until the scheduler is *between* actions, and keeps it that way for
        the body of the `with`. Use it to mutate state that a scheduled action reads or writes without
        racing that action: while held, no action is executing and the run loop can't start one.

        Implemented by taking `_execution_lock`, which the run loop holds for the full duration of each
        action. So don't call this from within a scheduled action: you'd block waiting for that action
        to finish, but it can't finish while it's stuck waiting here.

        (The clock system uses this so a tempo change made from a non-clock thread doesn't rewrite a
        clock's tempo while a scheduled action is in flight relying on it.)
        """
        with self._execution_lock:
            yield self

    def remove_events(self, matches: Callable[['QueueEvent'], bool]) -> int:
        """
        Remove all queued events that satisfy `matches(event)`. Returns the number removed.
        Used when a clock is killed and its pending wakeups (and pending forks) need to be cancelled.
        """
        with self._queue_change_condition:
            kept = [e for e in self._queue if not matches(e)]
            removed = len(self._queue) - len(kept)
            if removed:
                self._queue = kept
                heapq.heapify(self._queue)
                self._queue_change_condition.notify_all()
        return removed

    def reschedule(self, matches: Callable[['QueueEvent'], bool],
                   recompute: Callable[['QueueEvent'], float]) -> None:
        """
        Walk the heap and recompute `t` for any event that satisfies `matches(event)`.
        Used when a tempo change makes previously-scheduled wakeups stale.
        """
        with self._queue_change_condition:
            changed = False
            for i, event in enumerate(self._queue):
                if matches(event):
                    new_t = recompute(event)
                    if new_t != event.t:
                        self._queue[i] = QueueEvent(new_t, event.priority, event.action, event.metadata)
                        changed = True
            if changed:
                heapq.heapify(self._queue)
                self._queue_change_condition.notify_all()

    def schedule_action(self, t: float, action: Callable, priority: Tuple[int, ...] = (0,), metadata: Any = None) -> None:
        """
        Schedule the given action to be executed at time 't' (in seconds since scheduler start).
        """
        event = QueueEvent(t, priority, action, metadata)
        with self._queue_change_condition:
            heapq.heappush(self._queue, event)
            self._queue_change_condition.notify_all()
        logger.debug(f"Scheduled event '{metadata}' for time {t}")

    def run(self) -> None:
        """
        Main scheduler loop: wait until the next queued action is due, then execute it, forever.

        The two subtle parts — how the wait is structured around `_queue_change_condition` so a queue
        change can't be lost, and why execution is wrapped in `_execution_lock` — are explained inline
        at STEP 1 and STEP 2 below.
        """
        self._start_time = time.time()
        self._last_wake_time = self._start_time
        logger.info("Scheduler started")
        while not self._killed:
            # -------------------------   STEP 1: Wait for the next event -------------------------------
            # We hold the _queue_change_condition throughout so that nothing modifies the queue while we are
            # reading from it and calculating the wait time. Note that, since this is a Condition,
            # the underlying lock is released during self._queue_change_condition.wait(), allowing methods like
            # schedule_action, reschedule, and remove_events to modify the queue while we are waiting in between
            # events. Those methods then notify_all() on the condition, which wakes us from wait() to re-check
            # the now-updated queue.

            with self._queue_change_condition:
                # ~~~~~~ STEP 1a: If the queue is empty, wait for an event to be scheduled ~~~~~~
                # This wait is for when the scheduler is alive and nothing is scheduled. So we wait to be notified
                # that there has been some change to the _queue. Note that kill() also notifies _queue_change_condition,
                # which wakes us from this wait() and then breaks us out of the run loop after seeing self._killed=True
                while not self._queue and not self._killed:
                    self._queue_change_condition.wait()

                if self._killed:
                    break

                # ~~~~~~ STEP 1b: Peek at the queue, and calculate wait duration to next scheduled event ~~~~~~
                next_event = self._queue[0]
                now = time.time()
                relative_wait_dur = self._last_wake_time + (next_event.t - self._ideal_time) - now
                absolute_wait_dur = self._start_time + next_event.t - now
                wait_duration = self.timing_policy * relative_wait_dur + (1 - self.timing_policy) * absolute_wait_dur
                self._log_event_timing(next_event, relative_wait_dur, absolute_wait_dur, wait_duration)

                # ~~~~~~ STEP 1c: Wait for the next scheduled event (if in the future) ~~~~~~
                if wait_duration > 0:
                    # Note that since we've been holding _queue_change_condition, nothing can have changed about the
                    # queue since we calculated the wait duration. Calling wait releases the underlying lock so that
                    # functions can now modify the queue. If they do so before the wait_duration has played out,
                    # they notify _queue_change_condition, wake us up early, and we re-enter the loop so that we
                    # can recalculate based on the updated queue.
                    self._queue_change_condition.wait(timeout=wait_duration)
                    continue

            # NOTE: there is a gap here between waiting for the event time and performing the action, where we are
            # no longer holding _queue_change_condition. This leaves a window for an external thread to mutate the
            # queue, which is why we recheck validity at the start of _execute_event and bail if things changed.

            # --------------------------- STEP 2: Perform the scheduled action -------------------------------
            # Executed events happen under the _execution_lock. If an external thread wants to make sure that
            # we are not mid-action, it can use the context manager while_quiescent().
            #
            # In the context of the Clock system, the scheduled action parks the scheduler and allows the clock
            # to perform user code until the next wait. clock._reschedule_after_tempo_change uses while_quiescent()
            # to ensure that we only modify tempo/reschedule events once every clock is dormant again.
            with self._execution_lock:
                self._execute_event(next_event)
        logger.info("Scheduler terminated")

    def _log_event_timing(self, event: 'QueueEvent', rel_wait: float, abs_wait: float, actual_wait: float) -> None:
        """Detailed timing trace for one scheduled event (only formatted if DEBUG is enabled)."""
        if not logger.isEnabledFor(logging.DEBUG):
            return
        logger.debug(
            "event=%r ideal=%.6f rel_wait=%.6f abs_wait=%.6f blended_wait=%.6f",
            event.metadata, self._ideal_time, rel_wait, abs_wait, actual_wait,
        )

    def _execute_event(self, event: QueueEvent) -> None:
        """
        Remove the event from the queue and execute its action.
        Updates the scheduler's timing metrics accordingly.
        """
        # Confirm the head is still `event` before popping: between run()'s peek at the queue (Step 1b) and now,
        # the lock was released, so another thread may have pushed a smaller-t event or rescheduled one to the
        # front. If so, bail and let run() recompute timing against the new head — otherwise we'd pop the wrong
        # event and execute `event.action` with stale bookkeeping. We use the condition here purely as the queue
        # lock (no wait/notify), held only to pop.
        # The scheduled action then runs unguarded by _queue_change_condition; the mechanism for avoiding racing with
        # the action is to use scheduler.while_quiescent(), since _execution_lock is held around this whole call.
        with self._queue_change_condition:
            if self._queue and self._queue[0] == event:
                heapq.heappop(self._queue)
            else:
                return

        self._ideal_time = event.t
        logger.debug(f"Executing event '{event.metadata}' scheduled at {event.t}")
        try:
            event.action()
        except Exception as e:
            logger.exception(f"Error executing event '{event.metadata}': {e}")
        self._last_wake_time = time.time()

# Module-level scheduler instance.
_scheduler: Scheduler | None = None

def get_scheduler() -> Scheduler:
    """
    Returns a module-level scheduler instance, creating and starting it if necessary.
    """
    global _scheduler
    sched = _scheduler
    if sched is None or not sched.is_alive():
        sched = Scheduler(daemon=True)
        sched.start()
        _scheduler = sched
    return sched
