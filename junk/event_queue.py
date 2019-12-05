import time
import threading
from clockblocks.utilities import sleep_precisely, sleep_precisely_until
import logging


class EventQueue:

    def __init__(self, latency, max_sleep_proportion=0.4):
        """

        :param latency:
        :param max_sleep_proportion: what percent of the latency we allow ourselves to sleep before checking in for
            new events.
        """
        self.latency = latency
        self.max_sleep = max_sleep_proportion * self.latency
        self.queue = []
        self.wait_event = threading.Event()

    def add_to_queue(self, timestamp, function, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        self.queue.append((timestamp + self.latency, function, args, kwargs))
        self.queue.sort(key=lambda x: x[0])  # re-sort by time stamp

    def _run_thread(self):
        last_event_time = None
        while not self.wait_event.is_set():
            if len(self.queue) == 0:
                # don't bother with precise sleep here, since we're not actually trying to hit a particular event
                self.wait_event.wait(timeout=self.max_sleep)
            else:
                next_event_time, next_event_function, args, kwargs = self.queue[0]
                now = time.time()
                if next_event_time > now + self.max_sleep:
                    # no events for a bit; sleep a portion of the latency designated by max_sleep_proportion
                    sleep_precisely(self.max_sleep, self.wait_event)
                elif next_event_time >= now:
                    if next_event_time > now:
                        sleep_precisely_until(next_event_time, self.wait_event)

                    last_event_time = next_event_time
                    next_event_function(*args, **kwargs)
                    self.queue.pop(0)
                elif last_event_time is not None and next_event_time >= last_event_time:
                    # this will happen when we have an event either at the same time as another event or really soon
                    # after it. It will seem to be behind, since the last event took a moment to run, but really it
                    # was just the execution time of the last event.
                    if now - next_event_time > 0.01:
                        # if we're over 10ms behind, this is probably a perceptible issue, so we should warn that
                        # the events being scheduled (which should be more-or-less instantaneous) are taking too long
                        logging.warning("An event was delayed by > 10ms due to a long execution function. You are "
                                        "probably trying to run overly complex functions on the EventQueue.")
                    # otherwise just run the event
                    last_event_time = next_event_time
                    next_event_function(*args, **kwargs)
                    self.queue.pop(0)
                else:
                    # if we're here, then next_event_time < now
                    logging.warning("EventQueue encountering backlog! Increase latency to prevent inaccuracy.")
                    next_event_function(*args, **kwargs)
                    self.queue.pop(0)

    def run(self):
        threading.Thread(target=self._run_thread, daemon=True).start()

    def kill(self):
        self.wait_event.set()
