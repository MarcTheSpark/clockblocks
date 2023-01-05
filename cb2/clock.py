import threading
from cb2.tempo_envelope import TempoHistory
from cb2.scheduler import get_scheduler


class Clock:

    def __init__(self, name: str = None, parent: 'Clock' = None, initial_rate: float = None,
                 initial_tempo: float = None, initial_beat_length: float = None):
        self.name = name
        self.parent = parent

        # get the initial rate from whichever way it was set
        if initial_rate is initial_beat_length is initial_tempo is None:
            initial_rate = 1
        else:
            assert (initial_rate is None) + (initial_beat_length is None) + (initial_tempo is None) == 2, \
                "No more than one of initial_rate, initial_beat_length, and initial_tempo should be set."
            initial_rate = initial_rate if initial_rate is not None \
                else 1 / initial_beat_length if initial_beat_length is not None else initial_tempo / 60
        # tempo envelope, in seconds since I was created
        self.tempo_history = TempoHistory(initial_rate, units="rate")

        self._scheduler = get_scheduler()
        self._scheduler.hold()
        self._start_time_in_scheduler = self._scheduler.time()
        self._wait_event = threading.Event()

    ##################################################################################################################
    #                                        Boilerplate TempoHistory Functionality
    ##################################################################################################################

    def time(self) -> float:
        """
        How much time has passed since this clock was created.
        Either in seconds, if this is the master clock, or in beats in the parent clock, if this clock was the result
        of a call to fork.
        """
        return self.tempo_history.time()

    def beat(self) -> float:
        """
        How many beats have passed since this clock was created.
        """
        return self.tempo_history.beat()

    @property
    def beat_length(self) -> float:
        """
        The length of a beat in this clock in seconds.
        Note that beat_length, tempo and rate are interconnected properties, and that by setting one of them the
        other two are automatically set in response according to the relationship: beat_length = 1/rate = 60/tempo.
        Also, note that "seconds" refers to actual seconds only in the master clock; otherwise it refers to beats
        in the parent clock.
        """
        return self.tempo_history.beat_length

    @beat_length.setter
    def beat_length(self, b):
        self.tempo_history.beat_length = b

    @property
    def rate(self) -> float:
        """
        The rate of this clock in beats / second.
        Note that beat_length, tempo and rate are interconnected properties, and that by setting one of them the
        other two are automatically set in response according to the relationship: beat_length = 1/rate = 60/tempo.
        Also, note that "seconds" refers to actual seconds only in the master clock; otherwise it refers to beats
        in the parent clock.
        """
        return self.tempo_history.rate

    @rate.setter
    def rate(self, r):
        self.tempo_history.rate = r

    @property
    def tempo(self) -> float:
        """
        The rate of this clock in beats / minute
        Note that beat_length, tempo and rate are interconnected properties, and that by setting one of them the
        other two are automatically set in response according to the relationship: beat_length = 1/rate = 60/tempo.
        Also, note that "seconds" refers to actual seconds only in the master clock; otherwise it refers to beats
        in the parent clock.
        """
        return self.tempo_history.tempo

    @tempo.setter
    def tempo(self, t):
        self.tempo_history.tempo = t

    def wait(self, dt, units="beats"):
        beat_dur = dt if units == "beats" else self.tempo_history.get_beat_wait_from_time_wait(dt)
        time_dur = dt if units == "time" else self.tempo_history.get_wait_time(dt)
        wake_up_time = self.time() + time_dur

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
        # update tempo history
        self.tempo_history.advance(beat_dur)
