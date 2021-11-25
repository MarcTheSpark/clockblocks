import threading
import time
from utilities import sleep_precisely_until, current_clock


class Scheduler(threading.Thread):

    def __init__(self):
        super().__init__()
        self._queue = []
        self._hold_event = threading.Event()
        self._wait_event = threading.Event()
        self._start_time = None

    def run(self) -> None:
        self._start_time = time.time()
        while True:
            self._wait_event.clear()
            self._queue.sort()
            if len(self._queue) == 0:
                self._wait_event.wait(0.01)
            else:
                t, action_type, data = self._queue[0]
                sleep_precisely_until(t, self._wait_event)
                if self._wait_event.is_set():
                    # woken early
                    continue
                self._queue.pop(0)
                if action_type == "run_func":
                    data()
                elif action_type == "fork_func":

                    def wrapper():
                        Clock(self)
                        data()

                    threading.Thread(target=wrapper).start()
                elif action_type == "release":
                    data.set()
                    self._hold_event.wait()

    def fork(self, func):
        self._queue.append((time.time(), "fork_func", func))
        self._wait_event.set()


class Clock:

    def __init__(self, scheduler: Scheduler):
        self._start_time = None
        self._scheduler = scheduler
        self._hold_event = threading.Event()
        threading.current_thread().__clock__ = self

    def wait(self, dt):
        if self._start_time is None:
            self._start_time = time.time()
        self._scheduler._queue.append((time.time() + dt, "release", self._hold_event))
        self._scheduler._hold_event.set()
        self._hold_event.wait()
        self._hold_event.clear()



def clocko():
    print("hi")
    current_clock().wait(2)
    print("ho")
    current_clock().wait(4)
    print("he")


s = Scheduler()
s.start()
s.fork(clocko)
time.sleep(0.5)
s.fork(clocko)
time.sleep(0.5)
s.fork(clocko)
time.sleep(10)