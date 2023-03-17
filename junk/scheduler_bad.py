"""
IDEA: The scheduler is simple. It doesn't need to hold at a particular time.
Its main role is to process a queue of actions with assigned times, and apply the timing policy
to those actions.
As for the clock, the master clock has an action lock. Whenever the master clock is first created, it creates
and acquires that lock. Then, when it calls "wait", it releases that lock and sets a wakeup time in the scheduler.
When it wakes up, it acquires the lock again. Whenever any clock wakes up, it acquires the action lock from the
scheduler, so that only one clock can act at a time.
When a clock calls fork, it could:
- schedule the starting of the new thread and the creation of its clock, etc. for the current time in the scheduler,
so that it happens immediately after the next wait call.
- somehow start the new thread immediately and then wait for that thread to hit wait via a condition? Probably not
worth is.

Clock tempo change when clock is active?
Clock tempo change when clock is inactive, but there is an active clock?
Clock tempo change when no clocks are active

Could have an issue with getting accurate time stamps
"""


import threading
import time
from collections import namedtuple
from enum import Enum, auto
from typing import Any, Callable, Tuple
from cb2.utilities import sleep_precisely_until
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class QueueEvent:
    t: int
    action: Callable[[], None]
    priority: Tuple[int, ...]
    metadata: Any


class Stage(Enum):
    INACTIVE = auto()
    HOLDING= auto()
    WAITING = auto()
    PROCESSING = auto()
    ACTING = auto()
    KILLED = auto()


class SchedulerKilledException(Exception):
    pass


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
        # The event used to wait for the next scheduled action
        self._wait_event = threading.Event()
        # an event that another other thread uses to ensure that the scheduler has woken and updated
        self._updated_condition = threading.Condition()
        # Event used to hold the scheduler at a particular time, set by default so that no hold occurs
        self._hold_event = threading.Event()
        self._hold_event.set()
        # ensures that the queue cannot be accessed/modified by multiple threads simultaneously
        self._queue_lock = threading.Lock()
        self._current_stage = Stage.INACTIVE

    # --------------------------------------------- Status methods ----------------------------------------------

    def time(self, wake=False):
        """The idealized time in the scheduler"""
        if wake:
            self._wake_and_update()
        return self._ideal_time

    def wall_time(self):
        """The actual time that has passed since the start of the scheduler."""
        return time.time() - self._start_time

    @property
    def current_stage(self):
        return self._current_stage

    @property
    def alive(self):
        return self._current_stage not in (Stage.INACTIVE, Stage.KILLED)
    
    def kill(self):
        global _scheduler
        self._current_stage = Stage.KILLED
        if _scheduler is self:
            _scheduler = None

    def next_wakeup_time(self) -> float:
        """
        Returns the next scheduled action that the scheduler will wake up to perform (inf if there are none queued)
        """
        return float("inf") if len(self._queue) == 0 else self._queue[0].t

    # ------------------------------------------- Main Scheduling Loop -----------------------------------------------

    def run(self) -> None:
        self._start_time = self._last_wake_time = time.time()
        while self._current_stage != Stage.KILLED:
            logger.info("New scheduler cycle")
            with self._updated_condition:
                self._updated_condition.notifyAll()
            self._current_stage = Stage.HOLDING
            logger.debug("Scheduler hold phase")
            self._hold_event.wait()
            self._current_stage = Stage.PROCESSING
            logger.debug("Scheduler processing queue item")
            with self._queue_lock:
                logger.info(f"Processing queue: {self._queue}")
            self._process_next_queue_item()
            self._wait_event.clear()

    def _process_next_queue_item(self):
        """
        Looks at the next item in the queue, waits until it's scheduled time, and carries out its action
        If there is no item in the queue, waits forever.
        In either case, can be woken up early with a call to wake.
        """
        # process the next item in the queue
        if len(self._queue) == 0:
            logger.debug("No item in scheduler queue; waiting until event is added.")
            self._current_stage = Stage.WAITING
            # if there are no items in the queue, wait indefinitely until woken
            self._wait_event.wait()
            self._current_stage = Stage.PROCESSING
            logger.debug("Scheduler woken.")
            # ...then update the scheduler time based on when it was woken
            self._ideal_time = time.time() - self._start_time
            return

        # if there are items in the queue, we can assume they are sorted by time, so consider the first one
        queue_event = self._queue.pop(0)
        logger.info(f"Processing queue event {queue_event}")

        if queue_event.t < self._ideal_time:
            self._current_stage = Stage.ACTING
            logger.debug(f"Event scheduled for {queue_event.t}, which is in the past (current time is "
                         f"{self._ideal_time}). Performing action '{queue_event.metadata}' immediately.")
            queue_event.action()
            logger.debug(f"Done Performing action {queue_event.metadata}.")
            self._current_stage = Stage.PROCESSING
            return

        # dt is how much time should have passed since the last queued action
        dt = queue_event.t - self._ideal_time
        stop_sleeping_time = max(self._last_wake_time + dt * self.timing_policy, self._start_time + queue_event.t)
        self._current_stage = Stage.WAITING
        logger.debug(f"Waiting nominal {dt} in scheduler (actually {stop_sleeping_time - time.time()}).")
        sleep_precisely_until(stop_sleeping_time, self._wait_event)
        self._current_stage = Stage.PROCESSING
        # make note of when we woke up we'll try to stay true to this in the next wait
        if self._wait_event.is_set():
            # woken early, so update the time
            self._ideal_time = time.time() - self._start_time
            logger.debug(f"Scheduler woken early; updating time to {self._ideal_time}.")

            # and re-add the event to the queue, since we still need to wait for it
            self._schedule_queue_event(queue_event)
        else:
            # woken naturally because it reached the time at which the action should occur
            self._ideal_time = queue_event.t
            # note down when we woke, both in seconds since epoch, and in terms of scheduler time
            self._last_wake_time = time.time()
            self._current_stage = Stage.ACTING
            logger.debug(f"Scheduler waking at {self._ideal_time}. Performing action {queue_event.metadata}.")
            queue_event.action()
            logger.debug(f"Done Performing action {queue_event.metadata}.")
            self._current_stage = Stage.PROCESSING

    def _wake_and_update(self):
        # when called from the scheduler thread (or when the scheduler isn't running), do nothing, since not asleep
        if self.current_stage is not Stage.WAITING:
            # if the scheduler is not alive or is holding; no need to wake it.
            # likewise if this is being called from the scheduler thread (since it is surely in the middle of
            # performing an action and therefore already up-to-date)
            return

        # acquire the update condition, set the wait event to wake up the scheduler, and then wait for the scheduler
        # to notify that it has reached a new cycle and is therefore up-to-date.

        with self._updated_condition:
            # (note to future self: conditions use an underlying lock, which is acquired when we enter the context
            # manager here, and then released when `_updated_condition.wait()` is called, allowing the scheduler
            # thread to move forward, acquire the `_updated_condition` and notify that it is done. At that point,
            # this thread re-acquires the lock and then re-releases it when it exits the context manager.
            # Pretty flipping confusing, but the point is that this way, the scheduler thread can't race ahead
            # and notify the _updated_condition in between `_wait_event.set()` and `_updated_condition.wait()`,
            # because it needs to acquire the lock to do so, and the lock is only released when we call
            # `_updated_condition.wait()` below.)
            self._wait_event.set()
            self._updated_condition.wait()

    # ------------------------------------------- Holding/Releasing --------------------------------------------------

    def hold(self):
        logger.info(f"Calling hold from {threading.current_thread()}")
        # prepare the hold event
        self._hold_event.clear()
        # wake the scheduler so that it makes progresses to the holding stage
        self._wake_and_update()

    def release(self):
        logger.info(f"Calling release from {threading.current_thread()}")
        self._hold_event.set()

    # ---------------------------------------------- Scheduling ------------------------------------------------------

    def schedule_action(self, t, action: Callable, priority: Tuple[int, ...] = (0, ),  metadata: Any = None) -> None:
        """
        Schedule the given action to be called at the given time in the scheduler

        :param t: time (since start of the scheduler) when action should occur
        :param action: function to call
        :param priority: a tuple of integers, used to order actions scheduled at the same time
        :param metadata: any additional information. Note that events at the same time stamp will end up being sorted
            based on any inherent ordering of this parameter.
        """
        logger.debug(f"Scheduling Action '{metadata}'")
        self._schedule_queue_event(QueueEvent(t, action, priority, metadata))

    def _schedule_queue_event(self, queue_event: QueueEvent):
        with self._queue_lock:
            self._queue.append(queue_event)
            self._queue.sort(key=lambda qe: (qe.t, qe.priority))
        if queue_event.t <= self.next_wakeup_time():
            # if we're scheduling a new action before the next wake-up, we should wake
            # the scheduler so that it can change its planned wakeup
            self._wake_and_update()


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
