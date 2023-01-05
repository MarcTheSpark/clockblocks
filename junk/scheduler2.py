import threading
import time
from collections import namedtuple
from typing import Any, Callable
from cb2.utilities import sleep_precisely_until

QueueEvent = namedtuple("QueueEvent", "t action metadata")


class Scheduler(threading.Thread):

    def __init__(self, timing_policy=0.98, daemon=True):
        super().__init__(daemon=daemon)
        self._queue = []
        # start time of this scheduler (used for absolute timing policy)
        self._start_time = None
        # last wake time of this scheduler (used for relative timing policy)
        self._last_wake_time = None
        # the time from the scheduler's perspective
        self._ideal_time = 0
        self.timing_policy = timing_policy
        self._alive = False
        # The event used to wait for the next scheduled action
        self._wait_event = threading.Event()
        # an event that another other thread uses to ensure that the scheduler has woken and updated
        self._updated_event = threading.Event()
        # ensures that the queue cannot be accessed/modified by multiple threads simultaneously
        self._queue_lock = threading.Lock()

    def time(self):
        """The idealized time in the scheduler"""
        self._wake_and_update()
        return self._ideal_time

    def wall_time(self):
        """The actual time that has passed since the start of the scheduler."""
        return time.time() - self._start_time

    def start(self) -> None:
        self._start_time = self._last_wake_time = time.time()
        super().start()

    def run(self) -> None:
        self._alive = True
        while self._alive:
            self._updated_event.set()
            self._updated_event.clear()
            self.process_next_queue_item()

    def process_next_queue_item(self):
        """
        Looks at the next item in the queue, waits until it's scheduled time, and carries out its action
        If there is no item in the queue, waits forever.
        In either case, can be woken up early with a call to wake.
        """
        # process the next item in the queue
        if len(self._queue) == 0:
            # if there are no items in the queue, wait indefinitely until woken
            self._wait_event.wait()
            # ...then update the scheduler time based on when it was woken
            self._ideal_time = time.time() - self._start_time
            # ...and res    et the wait event, so it's ready to go again
            self._wait_event.clear()
        else:
            # if there are items in the queue, we can assume they are sorted by time, so consider the first one
            queue_event = self._queue.pop(0)

            if queue_event.t < self._ideal_time:
                queue_event.action()
                return

            # dt is how much time should have passed since the last queued action
            dt = queue_event.t - self._ideal_time
            stop_sleeping_time = max(self._last_wake_time + dt * self.timing_policy, self._start_time + queue_event.t)
            sleep_precisely_until(stop_sleeping_time, self._wait_event)

            # make note of when we woke up we'll try to stay true to this in the next wait
            if self._wait_event.is_set():
                # woken early, so update the time
                self._ideal_time = time.time() - self._start_time
                # and re-add the event to the queue, since we still need to wait for it
                self._schedule_queue_event(queue_event)
                # finally, clear the wait event so it continues to work
                self._wait_event.clear()
            else:
                # woken naturally because it reached the time at which the action should occur
                self._ideal_time = queue_event.t
                # note down when we woke, both in seconds since epoch, and in terms of scheduler time
                self._last_wake_time = time.time()
                queue_event.action()

    def _wake_and_update(self):
        # when called from the scheduler thread (or when the scheduler isn't running), do nothing, since not asleep
        # when called from another thread, wake up the scheduler thread, and then wait for it
        # to set the updated event, indicating that it has reoriented/updated.
        if self._alive and threading.current_thread() != self:
            self._wait_event.set()
            self._updated_event.wait()
            self._updated_event.clear()

    def next_wakeup_time(self) -> float:
        """
        Returns the next scheduled action that the scheduler will wake up to perform (inf if there are none queued)
        """
        return float("inf") if len(self._queue) == 0 else self._queue[0].t

    def _schedule_queue_event(self, queue_event: QueueEvent):
        with self._queue_lock:
            self._queue.append(queue_event)
            self._queue.sort(key=lambda qe: (qe.t, qe.metadata))
        if queue_event.t <= self.next_wakeup_time():
            # if we're scheduling a new action before the next wake-up, we should wake
            # the scheduler so that it can change its planned wakeup
            self._wake_and_update()

    def schedule_action(self, t, action: Callable, metadata: Any = None) -> None:
        """
        Schedule the given action to be called at the given time in the scheduler

        :param t: time (since start of the scheduler) when action should occur
        :param action: function to call
        :param metadata: any additional information. Note that events at the same time stamp will end up being sorted
            based on any inherent ordering of this parameter.
        """
        self._schedule_queue_event(QueueEvent(t, action, metadata))


_scheduler: Scheduler = None


def get_scheduler() -> Scheduler:
    """
    Returns a module-level scheduler object, creating a new one if needed.
    """
    global _scheduler
    if _scheduler is None:
        _scheduler = Scheduler(daemon=True)
        _scheduler.start()
    return _scheduler
