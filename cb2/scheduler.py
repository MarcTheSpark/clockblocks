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

        # _fast_forward_goal should be one of the following:
        #  1) None, meaning that we are not fast-forwarding; the normal state of the scheduler
        #  2) A scheduler time up until which we process events as fast as possible without sleeping, advancing
        #   _ideal_time as we go.
        #  3) float('inf), indicating that events will be processed as fast as possible until the fast forwarding
        #   state is turned off
        self._fast_forward_goal = None

        # When fast-forward ends we need to re-anchor the timing reference points (see _reanchor_timing), since
        # otherwise we would appear to be far ahead of schedule. There are two ways that fast-forward can end:
        #   (1) We reach a finite _fast_forward_goal. This is detected within _end_fast_forward_if_active when the
        #       next event lands at/beyond the goal, and _reanchor_timing is called directly. In this case the 
        #       _was_fast_forwarding flag below isn't used for detection, but it still must be cleared, because
        #       otherwise the next normal wait would think we had just been fast-forwarding and reanchor spuriously.
        #   (2) Fast forward was actively turned off (i.e. _fast_forward_goal was set back to None before it was
        #       reached). In this case, we need to know on the next run loop that we *were* fast_forwarding so
        #       that we can call _reanchor_timing. This is accomplished through the _was_fast_forwarding flag.
        #       Without this flag we would have no way of distinguishing between fast-forwarding having always
        #       been off, and fast-forwarding having just been turned off.
        # See _fast_forwarding_through (which sets the flag) and _end_fast_forward_if_active (which consumes it).
        self._was_fast_forwarding = False

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

    def set_fast_forward_goal(self, goal: float | None) -> None:
        """
        Set the fast-forward goal (a scheduler-time, float('inf') for indefinite, or None to stop) and
        wake the run loop so it acts on the change immediately rather than after its current timed wait.
        """
        # Mutated under _queue_change_condition to avoid interleaving with the run loop.
        with self._queue_change_condition:
            self._fast_forward_goal = goal
            self._queue_change_condition.notify_all()

    def is_fast_forwarding(self) -> bool:
        """Whether a fast-forward goal is currently set."""
        return self._fast_forward_goal is not None

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

        In the context of the clock system, it also shouldn't be called from the clock thread, since that
        will only be running while the scheduler is parked inside of _execute_event. For this reason we
        have :meth:`Clock.while_scheduler_quiescent`, which safely wraps this method by detecting whether
        we are running from a clock and no-op'ing in that case.
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
        logger.debug("Scheduled event %r for time %s", metadata, t)

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

                # Fast-forward short-circuits the wait. If we're fast-forwarding *through* this event (it
                # lies before the goal), fire it with no wall-clock delay. Otherwise we're about to wait in
                # real time, so first settle any fast-forward that was in progress — re-anchoring the timing
                # reference points — and then time the wait normally.
                if self._fast_forwarding_through(next_event):
                    self._was_fast_forwarding = True
                    wait_duration = 0.0
                else:
                    self._end_fast_forward_if_active(now)
                    wait_duration = self._compute_wait_duration(next_event, now)

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

    def _fast_forwarding_through(self, next_event: 'QueueEvent') -> bool:
        """
        Is a fast-forward in effect and does this event fall before the goal, so it should
        fire with no wall-clock wait?

        A None goal will always return False; we're not fast-forwarding. A float('inf') goal will always
        return True. Any other finite goal will return True if the next event is before the _fast_forward_goal.
        """
        return self._fast_forward_goal is not None and next_event.t < self._fast_forward_goal

    def _end_fast_forward_if_active(self, now: float) -> None:
        """
        Checks for and handles the transition from fast-forwarding to non-fast-forwarding state.
        Called from the run loop (while holding _queue_change_condition) whenever we are not fast_forwarding
        through the next event. Generally a no-op if we are not and have not been fast forwarding. The two
        cases where this function acts are:

          (1) When we have reached a finite _fast_forward_goal; in this case is_fast_forwarding() is still true,
            so we pin _ideal_time to the goal, turn off fast-forwarding (by setting _fast_forward_goal = None),
            clear _was_fast_forwarding (so that the next wait doesn't think we need to re-anchor), and
            _reanchor_timing to the current time.

          (2) When fast forwarding has been turned off externally. In this case, there's no _fast_forward_goal
            to read from or flip, so we're relying on _was_fast_forwarding. Flip that to false and _reanchor_timing.
        """
        if self.is_fast_forwarding():
            # we are fast-forwarding, but since this function was called, it means that we're about to stop
            # since the next event is set for *after* the _fast_forward_goal. Jump the _ideal_time straight
            # to the fast forward goal (making sure it's not backwards), clear the _fast_forward_goal,
            # and then reanchor timing
            self._ideal_time = max(self._ideal_time, self._fast_forward_goal)
            self._fast_forward_goal = None
            self._reanchor_timing(now)
            self._was_fast_forwarding = False
            return
        elif self._was_fast_forwarding:
            # We're not fast forwarding anymore, but last run iteration we were. This branch is only reached
            # when fast forwarding was canceled manually (by clearing self._fast_forward_goal); otherwise
            # we would have exited fast-forward via the block above.
            # in this case, all we need to do is reanchor the timing
            self._reanchor_timing(now)
            self._was_fast_forwarding = False
            return
        # if we neither are fast-forwarding nor were fast-forwarding, this is a normal non-fast-forwarding wait
        # return naturally as a no op

    def _reanchor_timing(self, now: float) -> None:
        """
        Reanchor the timing reference points after fast-forwarding:
            - _last_wake_time becomes now, so the next wait measures relative timing from when we stopped
                fast-forwarding.
            - self._start_time is set to self._ideal_time seconds in the past so that we are reanchored
                to be exactly on time as far as absolute timing is concerned.
        """
        self._last_wake_time = now
        self._start_time = now - self._ideal_time

    def _compute_wait_duration(self, next_event: 'QueueEvent', now: float) -> float:
        """Blend the relative- and absolute-timing wait durations per the timing policy (see __init__)."""
        relative_wait_dur = self._last_wake_time + (next_event.t - self._ideal_time) - now
        absolute_wait_dur = self._start_time + next_event.t - now
        wait_duration = self.timing_policy * relative_wait_dur + (1 - self.timing_policy) * absolute_wait_dur
        self._log_event_timing(next_event, relative_wait_dur, absolute_wait_dur, wait_duration)
        return wait_duration

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

        # Ideal time never runs backward. Events are popped in t-order, so normally event.t >= _ideal_time
        # and this is just `= event.t`; the max() guards the degenerate case of an event scheduled in the
        # past (a t that has already elapsed), which fires immediately but must not drag the scheduler's
        # clock backward — that would throw off the timing of everything reading _ideal_time.
        self._ideal_time = max(self._ideal_time, event.t)
        logger.debug("Executing event %r scheduled at %s", event.metadata, event.t)
        try:
            event.action()
        except Exception as e:
            logger.exception(f"Error executing event '{event.metadata}': {e}")
        self._last_wake_time = time.time()
