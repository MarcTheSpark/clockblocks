import time
import threading
from .utilities import sleep_precisely
import sched


# TODO:  Check if delayed. Why the hanging note?
class EventQueue:

    def __init__(self, latency=0, precise_sleep=True, running_behind_warning_threshold=0.01):
        """
        A queue to which we can add functions with timestamps, which then get executed in order.

        :param latency: optional amount by which time stamps are delayed when added to the queue
        :param precise_sleep: if true, use the "sleep_precisely" function, which incorporates a small busy wait
            at the end to try to be extra accurate.
        :param running_behind_warning_threshold: if a function call is happening more than this amount of seconds late,
            we throw a warning.
        """

        self.latency = latency
        self.scheduler = sched.scheduler(time.time, sleep_precisely if precise_sleep else time.sleep)
        self.running_behind_warning_threshold = running_behind_warning_threshold
        self.parent_thread = None

    def add_to_queue(self, timestamp, function, args=(), kwargs=None):
        self.scheduler.enterabs(timestamp + self.latency, 1, function, args, kwargs)

    def _run_thread(self):
        while True:
            if not self.scheduler.empty():
                self.scheduler.run()
            else:
                if self.parent_thread.is_alive():
                    time.sleep(self.latency / 2)
                else:
                    break

    def run(self):
        self.parent_thread = threading.current_thread()
        threading.Thread(target=self._run_thread, daemon=False).start()
