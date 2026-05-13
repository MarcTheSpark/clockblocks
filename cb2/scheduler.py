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
        self._queue_lock = threading.Lock()
        # Condition variable to notify when new events are scheduled.
        self._new_event = threading.Condition(self._queue_lock)
        # Hold/release mechanism. When cleared, event processing is paused.
        self._hold_event = threading.Event()
        self._hold_event.set()  # Not held by default.
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
            with self._new_event:
                self._new_event.notify_all()
        return self._ideal_time

    def wall_time(self) -> float:
        """Return the actual wall-clock time elapsed since the scheduler started."""
        return time.time() - self._start_time if self._start_time else 0.0

    def kill(self) -> None:
        """Stop the scheduler."""
        self._killed = True
        self._hold_event.set()  # unblock _wait_if_held so the loop can exit
        with self._new_event:
            self._new_event.notify_all()

    def _hold(self) -> None:
        """Pause the execution of scheduled events until released."""
        logger.debug("Scheduler hold activated")
        self._hold_event.clear()

    def _release(self) -> None:
        """Resume execution of scheduled events."""
        logger.debug("Scheduler hold released")
        self._hold_event.set()
        with self._new_event:
            self._new_event.notify_all()

    @contextmanager
    def held(self):
        """Context-manager wrapper around _hold()/_release(). Guarantees release on exception."""
        self._hold()
        try:
            yield self
        finally:
            self._release()

    def reschedule(self, matches: Callable[['QueueEvent'], bool],
                   recompute: Callable[['QueueEvent'], float]) -> None:
        """
        Walk the heap and recompute `t` for any event that satisfies `matches(event)`.
        Used when a tempo change makes previously-scheduled wakeups stale.
        """
        with self._new_event:
            changed = False
            for i, event in enumerate(self._queue):
                if matches(event):
                    new_t = recompute(event)
                    if new_t != event.t:
                        self._queue[i] = QueueEvent(new_t, event.priority, event.action, event.metadata)
                        changed = True
            if changed:
                heapq.heapify(self._queue)
                self._new_event.notify_all()

    def schedule_action(self, t: float, action: Callable, priority: Tuple[int, ...] = (0,), metadata: Any = None) -> None:
        """
        Schedule the given action to be executed at time 't' (in seconds since scheduler start).
        """
        event = QueueEvent(t, priority, action, metadata)
        with self._new_event:
            heapq.heappush(self._queue, event)
            self._new_event.notify_all()
        logger.debug(f"Scheduled event '{metadata}' for time {t}")

    def run(self) -> None:
        """Main scheduler loop, broken into helper methods for clarity."""
        self._start_time = time.time()
        self._last_wake_time = self._start_time
        logger.info("Scheduler started")
        while not self._killed:
            self._hold_event.wait()
            if self._killed:
                break
            next_event = self._get_next_event()
            if next_event is None:  # only happens on kill
                break

            now = time.time()
            relative_wait_dur = self._last_wake_time + (next_event.t - self._ideal_time) - now
            absolute_wait_dur = self._start_time + next_event.t - now
            wait_duration = self.timing_policy * relative_wait_dur + (1 - self.timing_policy) * absolute_wait_dur

            self._log_event_timing(next_event, relative_wait_dur, absolute_wait_dur, wait_duration)

            if wait_duration > 0:
                self._wait_for(wait_duration)
                continue  # Re-check the queue after waiting.
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

    def _get_next_event(self) -> QueueEvent | None:
        """
        Return the next event in the queue without removing it.
        If the queue is empty, wait until an event is scheduled (or the scheduler is killed).
        """
        with self._new_event:
            while not self._queue and not self._killed:
                self._new_event.wait()
            return self._queue[0] if self._queue else None

    def _wait_for(self, duration: float) -> None:
        """
        Wait for the specified duration or until a new event is scheduled.
        """
        with self._new_event:
            self._new_event.wait(timeout=duration)

    def _execute_event(self, event: QueueEvent) -> None:
        """
        Remove the event from the queue and execute its action.
        Updates the scheduler's timing metrics accordingly.
        """
        # Re-acquire the queue lock and confirm the head is still `event` before popping:
        # between _get_next_event peeking and now, another thread may have pushed a smaller-t
        # event or rescheduled one to the front. If so, bail and let run() recompute timing
        # against the new head — otherwise we'd pop the wrong event and execute `event.action`
        # with stale bookkeeping.
        # Note that this is not using the condition-specific abilities of _new_event, it's just
        # blocking actions (like schedule_action) that use those abilities
        with self._new_event:
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
