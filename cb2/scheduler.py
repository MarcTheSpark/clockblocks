import threading
import time
import heapq
from dataclasses import dataclass
from typing import Callable, Any, Tuple
import logging

logger = logging.getLogger(__name__)

@dataclass(order=True)
class QueueEvent:
    # Ordering is by t first, then by priority
    t: float
    priority: Tuple[int, ...]
    action: Callable[[], None]
    metadata: Any = None

class SchedulerKilledException(Exception):
    pass

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
        with self._new_event:
            self._new_event.notify_all()

    def hold(self) -> None:
        """Pause the execution of scheduled events until released."""
        logger.debug("Scheduler hold activated")
        self._hold_event.clear()

    def release(self) -> None:
        """Resume execution of scheduled events."""
        logger.debug("Scheduler hold released")
        self._hold_event.set()
        with self._new_event:
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
            self._wait_if_held()
            next_event = self._get_next_event()
            if next_event is None:
                continue

            dt = next_event.t - self._ideal_time
            now = time.time()
            relative_target_time = self._last_wake_time + dt
            absolute_target_time = self._start_time + next_event.t
            target_time = relative_target_time * self.timing_policy + absolute_target_time * (1 - self.timing_policy)
            wait_duration = target_time - now

            # print(next_event)
            # print(f"{self._ideal_time}")
            # print(f"{dt=}")
            # print(f"rel={relative_target_time - self._start_time}")
            # print(f"abs={absolute_target_time - self._start_time}")
            # print(f"tar={target_time - self._start_time}")
            # print(f"{wait_duration=}")

            if wait_duration > 0:
                self._wait_for(wait_duration)
                continue  # Re-check the queue after waiting.
            self._execute_event(next_event)
        logger.info("Scheduler terminated")

    def _wait_if_held(self) -> None:
        """Wait until the scheduler is released if currently held."""
        while not self._hold_event.is_set() and not self._killed:
            time.sleep(0.01)  # Avoid busy waiting.

    def _get_next_event(self) -> QueueEvent:
        """
        Return the next event in the queue without removing it.
        If the queue is empty, wait until an event is scheduled.
        """
        with self._new_event:
            while not self._queue and not self._killed:
                self._new_event.wait()
            if self._killed:
                return None
            return self._queue[0]

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
        with self._new_event:
            if self._queue and self._queue[0] == event:
                heapq.heappop(self._queue)
            else:
                return  # The queue changed; do nothing.
        self._ideal_time = event.t
        logger.debug(f"Executing event '{event.metadata}' scheduled at {event.t}")
        try:
            event.action()
        except Exception as e:
            logger.exception(f"Error executing event '{event.metadata}': {e}")
        self._last_wake_time = time.time()

# Module-level scheduler instance.
_scheduler: Scheduler = None

def get_scheduler() -> Scheduler:
    """
    Returns a module-level scheduler instance, creating and starting it if necessary.
    """
    global _scheduler
    if _scheduler is None or not _scheduler.is_alive():
        _scheduler = Scheduler(daemon=True)
        _scheduler.start()
    return _scheduler


# Example usage:
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    sched = get_scheduler()

    def my_action():
        print("Action executed at ideal time:", sched.time(), "wall time:", sched.wall_time())

    # Schedule an action 5 seconds after start.
    sched.schedule_action(5, my_action, metadata="Test Action")
    time.sleep(6)
    sched.kill()
