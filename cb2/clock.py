import math
import threading
import time
from cb2.scheduler import Scheduler, get_scheduler
from cb2.tempo_envelope import TempoHistory, TempoEnvelope
from utilities import wait


class Clock:

    def __init__(self, initial_rate: float = None, initial_tempo: float = None, initial_beat_length: float = None,
                 tempo_envelope: TempoEnvelope = None):
        # setup scheduler
        self.scheduler = get_scheduler()
        self._start_time_in_scheduler = self.scheduler.time()

        threading.current_thread().__clock__ = self
        self.tempo_history = Clock.setup_tempo_history(initial_rate, initial_tempo, initial_beat_length, tempo_envelope)
        self.rate = initial_rate
        self.wait_event = threading.Event()

    @staticmethod
    def setup_tempo_history(initial_rate, initial_tempo, initial_beat_length, tempo_envelope):
        if initial_rate is initial_tempo is initial_beat_length is tempo_envelope is None:
            # no tempo/rate/beat_length set, so set it to 60 bpm
            return TempoHistory(1, units="beatlength")
        else:
            if not (initial_rate is None) + (initial_beat_length is None) + \
                   (initial_tempo is None) + (tempo_envelope is None) == 3:
                raise ValueError("Only one of initial_rate, initial_beat_length, initial_tempo, or initial_"
                                 "tempo_envelope argument should be used.")
            if tempo_envelope is not None:
                # given a tempo_envelope to follow
                return TempoHistory.from_tempo_envelope(tempo_envelope)
            else:
                # otherwise, get the initial beat length
                if initial_rate is not None:
                    initial_beat_length = 1 / initial_rate
                elif initial_tempo is not None:
                    initial_beat_length = 60 / initial_tempo
                return TempoHistory(initial_beat_length, units="beatlength")

    def beat(self):
        return self.tempo_history.beat()

    def time(self):
        return self.tempo_history.time()

    def tempo(self):
        return self.tempo_history.tempo

    def wait(self, dt, units="beats"):
        if units == "beats":
            # wake_up_beat = self.beat() + dt  # TODO: NOT NEEDED?
            wake_up_time = self.time() + self.tempo_history.get_wait_time(dt)
        else:
            # wake_up_beat = self.beat() + self.tempo_history.get_beat_wait_from_time_wait(dt)   # TODO: NOT NEEDED?
            wake_up_time = self.time() + dt

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
        self.wait_event.wait()  # THIS IS WHERE OTHER THREADS TAKE OVER
        # hold the scheduler until the next wait call is made and wake-up is scheduled
        self.scheduler.hold()
        # update beat and time
        if units == "beats":
            self.tempo_history.advance(dt)
        else:
            self.tempo_history.advance_time(dt)

    def __getattr__(self, name):
        return getattr(self.tempo_history, name)


########################################## DEMOS ########################################


def scheduler_demo():
    scheduler = Scheduler(daemon=False)
    start = time.time()
    scheduler.start()
    scheduler.schedule_action(2, lambda: print(time.time() - start))
    time.sleep(1.0)
    scheduler.schedule_action(1.5, lambda: print("HERE"))
    scheduler.schedule_action(2.7, lambda: print(time.time() - start))
    scheduler.schedule_action(3, lambda: print(time.time() - start))



def timing_policy_demo():
    # simple function to print times
    scheduler = Scheduler(daemon=False)
    # scheduler.timing_policy = 0  # absolute timing policy
    # scheduler.timing_policy = 1  # relative
    # scheduler.timing_policy = 0.98  # compromise timing policy
    start = time.time()

    def print_time():
        print("scheduler time={}, actual time={}".format(scheduler.time(), time.time() - start))
    scheduler.start()
    # hold the scheduler and schedule a bunch of prints every second
    scheduler.hold()
    for t in range(1, 30):
        scheduler.schedule_action(t, print_time)
    # wait 1.5 seconds, which makes the first call 0.5 seconds behind
    time.sleep(1.5)
    # release the scheduler and watch it catch up (or not)
    scheduler.release()


TempoEnvelope.from_function(lambda b: 120 + math.sin(b) * 80, domain_end=100)
c = Clock()
c.apply_function(lambda b: 60 + (10*b) % 200)

while True:
    print(c.tempo_history)
    print(c.beat(), c.tempo())
    wait(1)

