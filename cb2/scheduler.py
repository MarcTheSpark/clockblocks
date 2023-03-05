import threading
import time
from collections import namedtuple
from enum import Enum, auto
from typing import Any, Callable
from cb2.utilities import sleep_precisely_until
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class QueueEvent:
    t: int
    action: Callable[[], None]
    metadata: Any


class Stage(Enum):
    INACTIVE = auto()
    HOLDING = auto()
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

    def time(self):
        """The idealized time in the scheduler"""
        self._wake_and_update()
        return self._ideal_time

    def wall_time(self):
        """The actual time that has passed since the start of the scheduler."""
        return time.time() - self._start_time

    @property
    def current_stage(self):
        return self._current_stage
    
    def _set_stage(self, stage: Stage):
        if self._current_stage == Stage.KILLED:
            raise SchedulerKilledException()
        self._current_stage = stage

    @property
    def alive(self):
        return self._current_stage not in (Stage.INACTIVE, Stage.KILLED)
    
    def kill(self):
        global _scheduler
        self._set_stage(Stage.KILLED)
        _scheduler = None

    def next_wakeup_time(self) -> float:
        """
        Returns the next scheduled action that the scheduler will wake up to perform (inf if there are none queued)
        """
        return float("inf") if len(self._queue) == 0 else self._queue[0].t

    # ------------------------------------------- Main Scheduling Loop -----------------------------------------------

    def run(self) -> None:
        self._start_time = self._last_wake_time = time.time()
        try:
            while self._current_stage != Stage.KILLED:
                logger.info("New scheduler cycle")
                with self._updated_condition:
                    self._updated_condition.notifyAll()
                self._set_stage(Stage.HOLDING)
                logger.debug("Scheduler holding")
                self._hold_event.wait()
                self._set_stage(Stage.PROCESSING)
                logger.debug("Scheduler processing queue item")
                self._process_next_queue_item()
        except SchedulerKilledException:
            pass

    def _process_next_queue_item(self):
        """
        Looks at the next item in the queue, waits until it's scheduled time, and carries out its action
        If there is no item in the queue, waits forever.
        In either case, can be woken up early with a call to wake.
        """
        # process the next item in the queue
        if len(self._queue) == 0:
            logger.debug("No item in scheduler queue; waiting until event is added.")
            self._set_stage(Stage.WAITING)
            # if there are no items in the queue, wait indefinitely until woken
            self._wait_event.wait()
            self._set_stage(Stage.PROCESSING)
            logger.debug("Scheduler woken.")
            # ...then update the scheduler time based on when it was woken
            self._ideal_time = time.time() - self._start_time
            # ...and res    et the wait event, so it's ready to go again
            self._wait_event.clear()
        else:
            # if there are items in the queue, we can assume they are sorted by time, so consider the first one
            queue_event = self._queue.pop(0)
            logger.debug(f"Processing queue event {queue_event}")

            if queue_event.t < self._ideal_time:
                self._set_stage(Stage.ACTING)
                logger.debug(f"Event scheduled for {queue_event.t}, which is in the past (current time is "
                             f"{self._ideal_time}). Performing action immediately.")
                queue_event.action()
                self._set_stage(Stage.PROCESSING)
                return

            # dt is how much time should have passed since the last queued action
            dt = queue_event.t - self._ideal_time
            stop_sleeping_time = max(self._last_wake_time + dt * self.timing_policy, self._start_time + queue_event.t)
            self._set_stage(Stage.WAITING)
            logger.debug(f"Waiting nominal {dt} in scheduler (actually {stop_sleeping_time - time.time()}).")
            sleep_precisely_until(stop_sleeping_time, self._wait_event)
            self._set_stage(Stage.PROCESSING)
            # make note of when we woke up we'll try to stay true to this in the next wait
            if self._wait_event.is_set():
                # woken early, so update the time
                self._ideal_time = time.time() - self._start_time
                logger.debug(f"Scheduler woken early; updating time to {self._ideal_time}.")

                # and re-add the event to the queue, since we still need to wait for it
                self._schedule_queue_event(queue_event)
                # finally, clear the wait event so it continues to work
                self._wait_event.clear()
            else:
                # woken naturally because it reached the time at which the action should occur
                self._ideal_time = queue_event.t
                # note down when we woke, both in seconds since epoch, and in terms of scheduler time
                self._last_wake_time = time.time()
                self._set_stage(Stage.ACTING)
                logger.debug(f"Scheduler woken at {self._ideal_time}. Performing action.")
                queue_event.action()
                self._set_stage(Stage.PROCESSING)

    def _wake_and_update(self):
        # when called from the scheduler thread (or when the scheduler isn't running), do nothing, since not asleep
        if not self.alive or self._current_stage == Stage.HOLDING or threading.current_thread() == self:
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
        # prepare the hold event
        self._hold_event.clear()
        # wake the scheduler so that it makes progresses to the holding stage
        self._wake_and_update()

    def release(self):
        self._hold_event.set()

    # ---------------------------------------------- Scheduling ------------------------------------------------------

    def schedule_action(self, t, action: Callable, metadata: Any = None) -> None:
        """
        Schedule the given action to be called at the given time in the scheduler

        :param t: time (since start of the scheduler) when action should occur
        :param action: function to call
        :param metadata: any additional information. Note that events at the same time stamp will end up being sorted
            based on any inherent ordering of this parameter.
        """
        self._schedule_queue_event(QueueEvent(t, action, metadata))

    def _schedule_queue_event(self, queue_event: QueueEvent):
        with self._queue_lock:
            self._queue.append(queue_event)
            self._queue.sort(key=lambda qe: (qe.t, qe.metadata))
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
