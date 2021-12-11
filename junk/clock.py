import threading
import time
from typing import Callable

from clockblocks import TempoEnvelope

from utilities import sleep_precisely_until, current_clock, wait


class Scheduler(threading.Thread):

    def __init__(self, daemon=True):
        super().__init__(daemon=daemon)
        self._queue = []
        self._wait_event = threading.Event()
        self._hold_event = threading.Event()
        self._ready_condition = threading.Condition()
        self._hold_event.set()
        # start time of this scheduler in seconds since epoch (result of time.time)
        self._start_time_sse = None
        # last time this scheduler awoke from sleep in seconds since epoch (result of time.time)
        self._last_wake_time_sse = None
        # last time the scheduler handled an action (in seconds since scheduler started)
        self._last_scheduled_action = 0
        self._timing_policy = 1
        self._t = 0

    def start(self) -> None:
        self._start_time_sse = self._last_wake_time_sse = time.time()
        super().start()

    def run(self) -> None:
        while True:
            # notify all other threads that we've reached a new cycle (so self.time() is up-to-date)
            print(threading.current_thread(), "at 1")
            with self._ready_condition:
                self._ready_condition.notify_all()
            print(threading.current_thread(), "at 2")
            # if another thread has called hold(), hold here until it calls release()
            self._hold_event.wait()
            print(threading.current_thread(), "at 3")
            # process the next item in the queue
            if len(self._queue) == 0:
                # if there are no items in the queue, wait indefinitely until woken
                self._wait_event.wait()
                # ...then update the scheduler time based on when it was woken
                self._t = time.time() - self._start_time_sse
                # ...and reset the wait event so it's ready to go again
                self._wait_event.clear()
            else:
                # if there are items in the queue, we can assume they are sorted by time, so consider the first one
                t, action = self._queue[0]

                # when we wait until depends on the timing policy
                # The timing policy is a float between 0 and 1, where 0 is absolute timing, and 1 is relative timing.
                # Relative timing means we try to keep the time since the last action as accurate as possible, while
                # Absolute timing policy means we try to keep to the time since creation as accurate as possible
                # The timing policy is the proportion of the relative time we are guaranteed to wait. So at 0.8,
                # we will wait until 80% of the relative timing or the absolute timing, whichever comes
                dt = t - self._last_scheduled_action
                # print((self._last_wake_time_sse + dt * self._timing_policy) % 1000, (self._start_time_sse + t) % 1000)
                stop_sleeping_time = max(self._last_wake_time_sse + dt * self._timing_policy, self._start_time_sse + t)

                sleep_precisely_until(stop_sleeping_time, self._wait_event)
                # make note of when we woke up we'll try to stay true to this in the next wait
                self._last_wake_time_sse = time.time()
                if self._wait_event.is_set():
                    # woken early, so do not pop the queue, we still need to wait for it
                    self._t = time.time() - self._start_time_sse
                    self._wait_event.clear()  # clear the wait event so it continues to work
                else:
                    # woken naturally because it reached the time at which the action should occur
                    self._queue.pop(0)
                    self._t = t
                    self._last_scheduled_action = t
                    action()
            print(threading.current_thread(), "at 4")

    def time(self):
        self.wake()
        return self._t

    def next_wakeup_time(self):
        return float("inf") if len(self._queue) == 0 else self._queue[0][0]

    def schedule_action(self, t, action: Callable) -> None:
        """
        Schedule the given action to be called at the given time in the scheduler

        :param t: time (since start of the scheduler) when action should occur
        :param action: function to call
        """
        self._queue.append((t, action))
        self._queue.sort()
        if t <= self.next_wakeup_time():
            # if we're scheduling a new action before the next wake-up, we should wake
            # the scheduler so that it can change its planned wakeup
            self.wake()

    def wait_until_ready(self):
        with self._ready_condition:
            self._ready_condition.wait()

    def wake(self) -> None:
        """
        Wake the scheduler, and wait until it has finished updating its time.
        """
        print("calling wake from ", threading.current_thread())
        self._wait_event.set()
        self.wait_until_ready()

    def hold(self):
        self._hold_event.clear()

    def wake_and_hold(self):
        self._hold_event.clear()
        self._wait_event.set()
        self.wait_until_ready()

    def release(self):
        self._hold_event.set()


_scheduler = None


def get_scheduler():
    global _scheduler
    if _scheduler is None:
        _scheduler = Scheduler(daemon=True)
        _scheduler.start()
    return _scheduler


class Clock:

    def __init__(self, initial_rate=1):
        self.scheduler = get_scheduler()
        self.scheduler.hold()
        print(self.scheduler.time())
        print(self.scheduler.time())
        print(self.scheduler.time())
        self._start_time_in_scheduler = self.scheduler.time()

        threading.current_thread().__clock__ = self
        self.beat = self.time = 0
        self.rate = initial_rate
        self.wait_event = threading.Event()

    def wait(self, dt, units="beats"):
        if units == "beats":
            wake_up_beat = self.beat + dt
            wake_up_time = self.time + dt / self.rate
        else:
            wake_up_beat = self.beat + dt * self.rate
            wake_up_time = self.beat + dt

        # clear the wait_event so that it will block
        self.wait_event.clear()
        # add the wake-up to the scheduler's queue
        self.scheduler.schedule_action(
            self._start_time_in_scheduler + wake_up_time,
            self.wait_event.set
        )
        # release the scheduler to process other actions
        self.scheduler.release()
        # wait to be woken up by the scheduler
        self.wait_event.wait()
        # hold the scheduler until the next wait call is made and wake-up is scheduled
        self.scheduler.hold()
        # update beat and time
        self.beat = wake_up_beat
        self.time = wake_up_time


scheduler = Scheduler(daemon=False)
start = time.time()
scheduler.start()

scheduler.schedule_action(2, lambda: print(time.time() - start))
scheduler.schedule_action(3, lambda: print(time.time() - start))



exit()

print("Creating scheduler, waiting 2 seconds")
get_scheduler()

time.sleep(2)

Clock()

# start = time.time()
# time.sleep(1.5)
# print("hi", time.time() - start, time.time() % 1000)
#
# wait(1)
# print("ho", time.time() - start, time.time() % 1000)
# wait(2)
# print("he", time.time() - start, time.time() % 1000)
