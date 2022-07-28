import math
import threading
import time
from cb2.scheduler import Scheduler, get_scheduler
from cb2.tempo_envelope import TempoHistory, TempoEnvelope
from utilities import wait


class Clock:

    def __init__(self, initial_rate: float = None, initial_tempo: float = None, initial_beat_length: float = None,
                 tempo_envelope: TempoEnvelope = None, parent: 'Clock' = None):
        # setup scheduler
        self._scheduler = get_scheduler()
        self._start_time_in_scheduler = self._scheduler.time()

        threading.current_thread().__clock__ = self
        self._tempo_history = Clock.setup_tempo_history(initial_rate, initial_tempo, initial_beat_length, tempo_envelope)
        self._wait_event = threading.Event()
        self.parent = parent

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

    ##################################################################################################################
    #                                        Boilerplate TempoHistory Functionality
    ##################################################################################################################

    def time(self) -> float:
        """
        How much time has passed since this clock was created.
        Either in seconds, if this is the master clock, or in beats in the parent clock, if this clock was the result
        of a call to fork.
        """
        return self._tempo_history.time()

    def beat(self) -> float:
        """
        How many beats have passed since this clock was created.
        """
        return self._tempo_history.beat()

    # def time_in_master(self) -> float:
    #     """
    #     How much time (in seconds) has passed since the master clock was created.
    #     """
    #     return self.master.time()

    @property
    def beat_length(self) -> float:
        """
        The length of a beat in this clock in seconds.
        Note that beat_length, tempo and rate are interconnected properties, and that by setting one of them the
        other two are automatically set in response according to the relationship: beat_length = 1/rate = 60/tempo.
        Also, note that "seconds" refers to actual seconds only in the master clock; otherwise it refers to beats
        in the parent clock.
        """
        return self._tempo_history.beat_length

    @beat_length.setter
    def beat_length(self, b):
        self._tempo_history.beat_length = b

    @property
    def rate(self) -> float:
        """
        The rate of this clock in beats / second.
        Note that beat_length, tempo and rate are interconnected properties, and that by setting one of them the
        other two are automatically set in response according to the relationship: beat_length = 1/rate = 60/tempo.
        Also, note that "seconds" refers to actual seconds only in the master clock; otherwise it refers to beats
        in the parent clock.
        """
        return self._tempo_history.rate

    @rate.setter
    def rate(self, r):
        self._tempo_history.rate = r

    @property
    def tempo(self) -> float:
        """
        The rate of this clock in beats / minute
        Note that beat_length, tempo and rate are interconnected properties, and that by setting one of them the
        other two are automatically set in response according to the relationship: beat_length = 1/rate = 60/tempo.
        Also, note that "seconds" refers to actual seconds only in the master clock; otherwise it refers to beats
        in the parent clock.
        """
        return self._tempo_history.tempo

    @tempo.setter
    def tempo(self, t):
        self._tempo_history.tempo = t

    ##################################################################################################################
    #                                                 Waiting
    ##################################################################################################################

    def wait(self, dt, units="beats"):
        if units == "beats":
            # wake_up_beat = self.beat() + dt  # TODO: NOT NEEDED?
            wake_up_time = self.time() + self._tempo_history.get_wait_time(dt)
        else:
            # wake_up_beat = self.beat() + self._tempo_history.get_beat_wait_from_time_wait(dt)   # TODO: NOT NEEDED?
            wake_up_time = self.time() + dt

        # clear the _wait_event so that it will block
        self._wait_event.clear()
        # add the wake-up to the _scheduler's queue
        self._scheduler.schedule_action(
            self._start_time_in_scheduler + wake_up_time,
            self._wait_event.set
        )
        # release the _scheduler to process other actions
        self._scheduler.release()
        # wait to be woken up by the _scheduler
        self._wait_event.wait()  # THIS IS WHERE OTHER THREADS TAKE OVER
        # hold the _scheduler until the next wait call is made and wake-up is scheduled
        self._scheduler.hold()
        # update beat and time
        if units == "beats":
            self._tempo_history.advance(dt)
        else:
            self._tempo_history.advance_time(dt)



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
c._tempo_history.apply_function(lambda b: 60 + (10 * b) % 200)

while True:
    print(c.beat(), c.tempo)
    wait(1)

