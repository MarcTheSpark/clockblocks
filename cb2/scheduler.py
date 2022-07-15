import threading
import time
from collections import namedtuple
from typing import Callable
from cb2.utilities import sleep_precisely_until


QueueEvent = namedtuple("QueueEvent", "t action")


class Scheduler(threading.Thread):

    def __init__(self, timing_policy=0.98, daemon=True):
        super().__init__(daemon=daemon)
        self._queue = []
        self._wait_event = threading.Event()
        self._hold_event = threading.Event()
        self._ready_condition = threading.Condition()
        self._hold_event.set()
        # start time of this scheduler in seconds since epoch (result of time.time)
        self._start_time = None
        # last time this scheduler awoke from sleep in seconds since epoch (result of time.time)
        self._last_wake_time = None
        self.timing_policy = 0.98
        self._t = 0
        self._holding = False
        self._performing_scheduled_action = False
        self._alive = False

    def start(self) -> None:
        self._start_time = self._last_wake_time = time.time()
        super().start()

    def run(self) -> None:
        self._alive = True
        while self._alive:
            # if another thread has called hold(), hold here until it calls release()
            # 1) Hold phase
            if not self._hold_event.is_set():
                self._holding = True
                self._hold_event.wait()
                self._holding = False
            # 2) processing queue item phase
            self.process_next_queue_item()
            # notify all other threads that we've reached a new cycle (so self.time() is up-to-date)
            with self._ready_condition:
                self._ready_condition.notify_all()

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
            self._t = time.time() - self._start_time
            # ...and reset the wait event, so it's ready to go again
            self._wait_event.clear()
        else:
            # if there are items in the queue, we can assume they are sorted by time, so consider the first one
            t, action = self._queue[0]

            # dt is how much time should have passed since the last queued action
            dt = t - self._t
            stop_sleeping_time = max(self._last_wake_time + dt * self.timing_policy, self._start_time + t)
            sleep_precisely_until(stop_sleeping_time, self._wait_event)

            # make note of when we woke up we'll try to stay true to this in the next wait
            if self._wait_event.is_set():
                # woken early, so do not pop the queue, we still need to wait for it
                self._t = time.time() - self._start_time
                self._wait_event.clear()  # clear the wait event so it continues to work
            else:
                # woken naturally because it reached the time at which the action should occur
                self._queue.pop(0)
                self._t = t
                # note down when we woke, both in seconds since epoch, and in terms of scheduler time
                self._last_wake_time = time.time()
                self._performing_scheduled_action = True
                action()
                self._performing_scheduled_action = False

    def time(self) -> float:
        """
        Returns the current time in the scheduler.
        """
        self.wake()
        return self._t

    def next_wakeup_time(self) -> float:
        """
        Returns the next scheduled action that the scheduler will wake up to perform (inf if there are none queued)
        """
        return float("inf") if len(self._queue) == 0 else self._queue[0].t

    def schedule_action(self, t, action: Callable) -> None:
        """
        Schedule the given action to be called at the given time in the scheduler

        :param t: time (since start of the scheduler) when action should occur
        :param action: function to call
        """
        self._queue.append(QueueEvent(t, action))
        self._queue.sort()
        if t <= self.next_wakeup_time():
            # if we're scheduling a new action before the next wake-up, we should wake
            # the scheduler so that it can change its planned wakeup
            self.wake()

    def wake(self) -> None:
        """
        Wake the scheduler, and wait until it has finished updating its time.
        If the scheduler is holding, we consider it already awake.
        """
        if not self._holding and not self._performing_scheduled_action:
            # exit the wait
            self._wait_event.set()
            # ...and then allow the scheduler to catch up and update its time before returning
            with self._ready_condition:
                self._ready_condition.wait()

    def wake_and_hold(self):
        # we clear the hold event first so that the scheduler can't accidentally make it past before we clear it
        self.hold()
        self.wake()

    def hold(self):
        self._hold_event.clear()

    def release(self):
        if self._holding:
            self._hold_event.set()

    def kill(self):
        self.release()
        self.wake()
        self._alive = False
        self.join()


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
