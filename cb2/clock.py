import logging
import threading
import time
from numbers import Real
from typing import Callable, Sequence, Union

from cb2.tempo_envelope import TempoHistory
from cb2.scheduler import get_scheduler, Scheduler
from cb2.metric_phase import MetricPhaseTarget


class Clock:

    def __init__(self, name: str = None, parent: 'Clock' = None, initial_rate: float = None,
                 initial_tempo: float = None, initial_beat_length: float = None, scheduler: Scheduler = None):
        self.name = name
        self.parent = parent

        if self.is_master():
            threading.current_thread().__clock__ = self
            self.parent_offset = 0
        else:
            self.parent_offset = self.parent.beat()

        self._children = []

        # tempo envelope, in seconds since I was created
        self.tempo_history = TempoHistory(
            Clock._rate_tempo_or_beat_length_to_rate(initial_rate, initial_tempo, initial_beat_length),
            units="rate"
        )

        # get a default (shared) scheduler unless one is specifically provided
        self.scheduler = get_scheduler() if scheduler is None else scheduler
        self.scheduler.hold()
        self._start_time_in_scheduler = self.scheduler.time()
        self._wait_event = threading.Event()
        self._start_beat_in_parent = 0

    @staticmethod
    def _rate_tempo_or_beat_length_to_rate(rate, tempo, beat_length) -> float:
        if rate is tempo is beat_length is None:
            return 1
        if not (rate is None) + (tempo is None) + (beat_length is None) == 2:
            # exactly one of the arguments must be non-None
            raise ValueError("No more than one of `rate`, `tempo`, or `beat_length` may be defined.")
        return rate if rate is not None else 1 / beat_length if beat_length is not None else tempo / 60

    ##################################################################################################################
    #                                                     Properties
    ##################################################################################################################

    @property
    def master(self) -> 'Clock':
        """
        The master clock under which this clock operates (possibly itself)
        """
        return self if self.is_master() else self.parent.master

    def is_master(self) -> bool:
        """
        Check if this is the master clock

        :return: True if this is the master clock, False otherwise
        """
        return self.parent is None

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

    def wall_time_in_scheduler(self) -> float:
        """
        How long has this clock been alive in the scheduler. Should return a result very close to `Clock.time`
        """
        return self.scheduler.wall_time()  - self._start_time_in_scheduler

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

    def get_dt_in_scheduler(self, dt):
        """
        Gets the amount of time that should pass in the scheduler for a given time (not beats) in this clock.

        :param dt: how much time passes in this clock (and therefore beats in parent clock, if it exists)
        :return: how much time should pass in the scheduler
        """
        if self.parent is None:
            return dt
        else:
            return self.parent.tempo_history.integrate_interval(
                self.parent_offset + self.time(),
                self.parent_offset + self.time() + dt
            )

    def wait(self, dt, units="beats"):
        beat_dur = dt if units == "beats" else self.tempo_history.get_beat_wait_from_time_wait(dt)
        time_dur = dt if units == "time" else self.tempo_history.get_wait_time(dt)
        wake_up_time = self.time() + self.get_dt_in_scheduler(time_dur)

        # clear the _wait_event so that it will block
        self._wait_event.clear()
        # add the wake-up to the scheduler's queue
        self.scheduler.schedule_action(
            self._start_time_in_scheduler + wake_up_time,
            self._wait_event.set,
        )
        # release the scheduler to process other actions
        self.scheduler.release()
        # wait to be woken up by the scheduler
        self._wait_event.wait()  # THIS IS WHERE OTHER THREADS TAKE OVER
        # hold the scheduler until the next wait call is made and wake-up is scheduled
        self.scheduler.hold()
        # update tempo history
        self.tempo_history.advance(beat_dur)

    def fork(self, process_function: Callable, args: Sequence = (), kwargs: dict = None, name: str = None,
             initial_rate: float = None, initial_tempo: float = None, initial_beat_length: float = None,
             schedule_at: Union[float, MetricPhaseTarget] = None, done_callback: Callable[[], None] = None):
        """
        Spawns a parallel process running on a child clock.

        :param process_function: function defining the process to be spawned
        :param args: arguments to be passed to the process function. One subtlety to note here: if the number of
            arguments passed is one fewer than the number taken by the function, the clock on which the process is
            forked will be passed as the first argument, followed by the arguments given. For instance, if we define
            "forked_function(clock, a, b)", and then call "parent.fork(forked_function, (13, 6))", 13 will be passed
            to "a" and 6 to "b", while the clock on which forked_function is running will be passed to "clock". On the
            other hand, if the signature of the function were "forked_function(a, b)", 13 would be simply be passed to
            "a" and 6 to "b".
        :param kwargs: keyword arguments to be passed to the process function
        :param name: name to be given to the clock of the spawned child process
        :param initial_rate: starting rate of this clock (if set, don't set initial tempo or beat length)
        :param initial_tempo: starting tempo of this clock (if set, don't set initial rate or beat length)
        :param initial_beat_length: starting beat length of this clock (if set, don't set initial tempo or rate)
        :param schedule_at: either a beat or a :class:`~clockblocks.tempo_envelope.MetricPhaseTarget` specifying when we
            want this forked process to begin. The default value of None indicates that it is to begin immediately. A
            float indicates the beat in this clock at which the process is to start (should be in the future).
            Alternatively, a MetricPhaseTarget can be used to specify where in a regular cycle the process should begin.
            For instance, if we want to sync every fork to 3/4 time, MetricPhaseTarget(0, 3) would start a process on
            the downbeat, MetricPhaseTarget(1, 3) would start it on beat 2, and MetricPhaseTarget(2.5, 3) would start it
            halfway through beat 3.
        :param done_callback: a callback function to be invoked when the clock has terminated
        :return: the clock of the spawned child process
        """
        name = (process_function.__name__ if hasattr(process_function, '__name__') else "UNNAMED") \
            if name is None else name

        child = Clock(name, parent=self, initial_rate=initial_rate, initial_tempo=initial_tempo,
                      initial_beat_length=initial_beat_length)
        self._children.append(child)

        # if schedule_at is None:
        #     start_delay = 0
        # elif isinstance(schedule_at, Real):
        #     start_delay = schedule_at - self.beat()
        #     if start_delay < 0:
        #         logging.warning("`schedule_at` argument specified a beat in the past; forking immediately.")
        #         start_delay = 0
        # else:  # it's a MetricPhaseTarget
        #     if not isinstance(schedule_at, MetricPhaseTarget):
        #         raise ValueError("`schedule_at` must be either a float or a MetricPhaseTarget")
        #     # get_nearest_matching_beats returns the nearest match below and above, in order of nearness
        #     # we want the match above, since it's in the future, so we use max
        #     start_delay = max(*schedule_at.get_nearest_matching_beats(self.beat())) - self.beat()


        def _process(*args, **kwds):
            # set the implicit variable __clock__ in this thread
            threading.current_thread().__clock__ = child

            """
            The whole function we are forking is wrapped in a try/except clause, because we want to be able to kill
            it at will. When and if "kill" is called on the clock, its wait_event is set free and it immediately
            raises a ClockKilledError, which exits us from the process. (It's also possible, but unlikely, that
            we will get a DeadClockError, if we were just in the process of calling wait.)
            """
            # # Adjust for start delay
            # if start_delay > 0:
            #     # if there's a start delay, then we start the clock on a negative beat and time
            #     # so that both arrive at zero when the forked process starts
            #     child.tempo_history._t = -start_delay
            #     # child.tempo_history.segments[0].start_level is the initial beat length, so this
            #     # modifies the start beat proportionally to arrive at zero
            #     child.tempo_history._beat = -start_delay / child.tempo_history.segments[0].start_level
            #     child.parent_offset += start_delay
            #     child.wait(start_delay, units="time")

            # Run the function
            process_function(*args, **kwds)

            # Remove this sub-clock from the children list of the forking parent (self is the parent)
            self._children.remove(child)

            child._killed = True

            if done_callback is not None:
                done_callback()

        # self._run_in_pool(_process, args, kwargs)
        self.scheduler.schedule_action(
            self.scheduler.time(),
            threading.Thread(target=_process, args=args, kwargs=kwargs, daemon=True).start
        )

    def __repr__(self):
        child_list = "" if len(self._children) == 0 else ", ".join(str(child) for child in self._children)
        return ("Clock('{}')".format(self.name) if self.name is not None else "UNNAMED") + "[" + child_list + "]"