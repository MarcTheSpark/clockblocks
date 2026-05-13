import functools
import logging
import threading
from itertools import count
from numbers import Real
from typing import Callable, Sequence, Union, Iterator
from cb2.tempo_envelope import TempoHistory
from cb2.scheduler import get_scheduler, Scheduler
from cb2.metric_phase import MetricPhaseTarget
from cb2.enums import DurationUnits
from cb2.utilities import _PrintColors
import textwrap


def _reschedule_after_tempo_change(fn):
    """
    Decorator for Clock tempo/rate/beat_length setters. Holds the scheduler, brings the target clock's
    tempo_history up to the current scheduler position (so the change applies forward, not retroactively
    from the last committed beat), applies the mutation, then recomputes the scheduler-time of any queued
    wakeups whose `acting_clock` is self or a descendant.
    """
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with self.scheduler.held():
            self.bring_up_to_date()
            result = fn(self, *args, **kwargs)
            self._reschedule_self_and_descendants()
        return result
    return wrapper


class Clock:

    def __init__(self, name: str = None, parent: 'Clock' = None, initial_rate: float = None,
                 initial_tempo: float = None, initial_beat_length: float = None, scheduler: Scheduler = None):
        self.name = name
        self.parent = parent
        self._children = []
        self.set_id()

        # tempo envelope, in seconds since I was created
        self.tempo_history = TempoHistory(
            Clock._rate_tempo_or_beat_length_to_rate(initial_rate, initial_tempo, initial_beat_length),
            units="rate"
        )

        # get a default (shared) scheduler unless one is specifically provided
        self.scheduler = self.parent.scheduler if self.parent is not None else \
            get_scheduler() if scheduler is None else scheduler

        self._entering_wait_condition = threading.Condition()
        self._wait_event = threading.Event()

        if self.is_master():
            # the first thing we do is stop and put things in the scheduler's hands
            # tell the scheduler to wake up right away and get this clock going, then wait for the scheduler to do it
            self.scheduler.schedule_action(self.scheduler.time(), self._wake_and_advance_to_next_wait,
                                           (0, ), {"description": f"Initial wake for {self}", "acting_clock": self})
            self._wait_event.wait()

            threading.current_thread().__clock__ = self
            self.parent_offset = self._start_time_in_scheduler = self.scheduler.time()
        else:
            self.parent_offset = self.parent.beat()
            self._start_time_in_scheduler = self.parent_offset + self.parent._start_time_in_scheduler

    def set_id(self):
        self._child_counter = count()
        if self.is_master():
            self.clock_id = 0,
        else:
            self.clock_id = self.parent.clock_id + (next(self._child_counter), )

    @staticmethod
    def _rate_tempo_or_beat_length_to_rate(rate, tempo, beat_length) -> float:
        if rate is tempo is beat_length is None:
            return 1
        if not (rate is None) + (tempo is None) + (beat_length is None) == 2:
            # exactly one of the arguments must be non-None
            raise ValueError("No more than one of `rate`, `tempo`, or `beat_length` may be defined.")
        return rate if rate is not None else 1 / beat_length if beat_length is not None else tempo / 60

    ##################################################################################################################
    #                                                 Family Matters
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

    def children(self) -> Sequence['Clock']:
        """
        Get all direct child clocks forked by this clock.

        :return: tuple of all child clocks of this clock
        """
        return tuple(self._children)

    def iterate_inheritance(self, include_self: bool = True) -> Iterator['Clock']:
        """
        Iterate through parent, grandparent, etc. of this clock up until the master clock

        :param include_self: whether or not to include this clock in the iterator or start with the parent
        :return: iterator going up the clock family tree up to the master clock
        """

        clock = self
        if include_self:
            yield clock
        while clock.parent is not None:
            clock = clock.parent
            yield clock

    def inheritance(self, include_self: bool = True) -> Sequence['Clock']:
        """
        Get all parent, grandparent, etc. of this clock up until the master clock

        :param include_self: whether or not to include this clock in the iterator or start with the parent
        :return: tuple containing the clock's inheritance
        """
        return tuple(self.iterate_inheritance(include_self))

    def iterate_all_relatives(self, include_self: bool = False) -> Iterator['Clock']:
        """
        Iterate through all related clocks to this clock.

        :param include_self: whether or not to include this clock in the iterator
        :return: iterator going through all clocks in the family tree, starting with the master
        """
        if include_self:
            return self.master.iterate_descendants(True)
        else:
            return (c for c in self.master.iterate_descendants(True) if c is not self)

    def iterate_descendants(self, include_self: bool = False) -> Iterator['Clock']:
        """
        Iterate through all children, grandchildren, etc. of this clock

        :param include_self: whether or not to include this clock in the iterator
        :return: iterator going through all descendants
        """
        if include_self:
            yield self
        for child_clock in self._children:
            yield child_clock
            for descendant_of_child in child_clock.iterate_descendants():
                yield descendant_of_child

    def descendants(self) -> Sequence['Clock']:
        """
        Get all children, grandchildren, etc. of this clock

        :return: tuple of all descendants
        """
        return tuple(self.iterate_descendants())

    def print_family_tree(self) -> None:
        """
        Print a hierarchical representation of this clock's family tree.
        """
        print(self.master._child_tree_string(self))

    def _child_tree_string(self, highlight_clock: 'Clock' = None) -> str:
        name_text = self.name if self.name is not None else "(UNNAMED)"
        if highlight_clock is self:
            name_text = _PrintColors.BOLD + name_text + _PrintColors.END
        children = self.children()
        if len(children) == 0:
            return name_text
        return "{}:\n{}".format(
            name_text,
            textwrap.indent("\n".join(child._child_tree_string(highlight_clock) for child in self.children()), "  ")
        )

    ##################################################################################################################
    #                                        Boilerplate TempoHistory Functionality
    ##################################################################################################################

    def time(self) -> float:
        """
        How much time has passed since this clock was created.
        Either in seconds, if this is the master clock, or in beats in the parent clock, if this clock was the result
        of a call to fork.

        Computed lazily from the current scheduler position, so it stays correct when called from any thread —
        including while the owning thread is mid-wait. Does not mutate `tempo_history` (the committed pointer).
        """
        return self.scheduler_to_clock_time(self.scheduler.time(), desired_units="time")

    def beat(self) -> float:
        """
        How many beats have passed since this clock was created. See `time()` for thread/laziness semantics.
        """
        return self.scheduler_to_clock_time(self.scheduler.time(), desired_units="beats")

    def wall_time_in_scheduler(self) -> float:
        """
        How long has this clock been alive in the scheduler. Should return a result very close to `Clock.time`
        """
        return self.scheduler.wall_time() - self._start_time_in_scheduler

    def status(self, verbose: bool = False) -> str:
        """
        A snapshot of this clock's name, beat, time, and wall time in the scheduler.
        """
        name = self.name if self.name is not None else "UNNAMED"
        if verbose:
            return (f"Clock {name!r}\n"
                    f"  beat: {self.beat():.9f}\n"
                    f"  time: {self.time():.9f}\n"
                    f"  wall: {self.wall_time_in_scheduler():.9f}")
        return f"[{name} beat={self.beat():.3f} time={self.time():.3f} wall={self.wall_time_in_scheduler():.3f}]"

    def print_status(self, verbose: bool = False) -> None:
        print(self.status(verbose), flush=True)

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
    @_reschedule_after_tempo_change
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
    @_reschedule_after_tempo_change
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
    @_reschedule_after_tempo_change
    def tempo(self, t):
        self.tempo_history.tempo = t

    def clock_to_scheduler_time(self, beat_or_time, units="beats"):
        """
        Gets the time in the scheduler for a given beat or time in this clock, working recursively up the chain
        of clocks.

        :param beat_or_time: the beat or time of interest in this clock
        :param units: one of ("beats", "time"), determining the units for the first argument
        :return: how much time should pass in the scheduler
        """
        units = DurationUnits(units)
        t = (beat_or_time if units == "time" else self.tempo_history.time_at_beat(beat_or_time)) + self.parent_offset
        for clock in self.inheritance(include_self=False):
            t = clock.tempo_history.time_at_beat(t) + clock.parent_offset
        return t

    def scheduler_to_clock_time(self, scheduler_time, desired_units="beats"):
        """
        Gets beats or time in this clock for a given time in the scheduler.

        :param scheduler_time: the time of interest in the scheduler
        :param desired_units: one of ("beats", "time"), whether we're looking for beats or time in this clock
        :return: how much time should pass in the scheduler
        """
        desired_units = DurationUnits(desired_units)
        t = scheduler_time
        for clock in reversed(self.inheritance(include_self=False)):
            t = clock.tempo_history.beat_at_time(t - clock.parent_offset)
        if desired_units == DurationUnits.TIME:
            return t - self.parent_offset
        else:
            return self.tempo_history.beat_at_time(t - self.parent_offset)

    def bring_up_to_date(self):
        """
        Advance this clock's tempo_history to match where it currently is in scheduler time.

        Called before mutating tempo from a foreign thread: if we don't, the tempo setter's `truncate()`
        cuts at the last *committed* beat (whatever the clock last woke at), causing the new tempo to
        retroactively reshape the segment the clock is currently napping through. From the owning thread
        this is a no-op since committed == current.
        """
        # delta = (live scheduler-derived beat) - (last committed beat in tempo_history)
        delta = self.beat() - self.tempo_history.beat()
        if delta > 0:
            self.tempo_history.advance(delta)

    def _reschedule_self_and_descendants(self):
        """
        After a tempo change on self, any queued scheduler-event whose `acting_clock` is self or a
        descendant of self has a stale `t` (computed against the old tempo curve). Recompute via
        clock_to_scheduler_time(target_beat) for each match.
        """
        affected = {id(c) for c in self.iterate_descendants(include_self=True)}

        def matches(event):
            meta = event.metadata
            if not isinstance(meta, dict):
                return False
            ac = meta.get("acting_clock")
            return ac is not None and id(ac) in affected and "target_beat" in meta

        def recompute(event):
            meta = event.metadata
            return meta["acting_clock"].clock_to_scheduler_time(meta["target_beat"])

        self.scheduler.reschedule(matches, recompute)

    ##################################################################################################################
    #                                              Waiting and Forking
    ##################################################################################################################

    def wait(self, dt, units="beats"):
        units = DurationUnits(units)
        if units == DurationUnits.BEATS:
            wake_up_beat = self.beat() + dt
            wake_up_time = self.tempo_history.time_at_beat(wake_up_beat)
        else:
            wake_up_time = self.time() + dt
            wake_up_beat = self.tempo_history.beat_at_time(wake_up_time)
        wake_up_time_in_scheduler = self.clock_to_scheduler_time(wake_up_time, units="time")

        # clear the _wait_event so that it will block
        self._wait_event.clear()

        # add the wake-up to the scheduler's queue
        self.scheduler.schedule_action(
            wake_up_time_in_scheduler,
            self._wake_and_advance_to_next_wait,
            self.clock_id,
            {"description": f"{self} wakeup action",
             "acting_clock": self,
             "target_beat": wake_up_beat}
        )

        with self._entering_wait_condition:
            # wait to be woken up by the scheduler
            self._entering_wait_condition.notifyAll()

        self._wait_event.wait()  # THIS IS WHERE OTHER THREADS TAKE OVER

        # advance the committed pointer of tempo_history from where it was to where we just woke up.
        # Use tempo_history.beat() directly (not self.beat(), which is the live scheduler-derived position).
        self.tempo_history.advance(wake_up_beat - self.tempo_history.beat())

    def _wake_and_advance_to_next_wait(self):
        """
        This function is scheduled on the scheduler: it wakes up this clock, and then blocks until the clock
        has hit another wait call and scheduled its next wakeup.
        """
        with self._entering_wait_condition:
            self._wait_event.set()
            self._entering_wait_condition.wait()

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

        if schedule_at is None:
            start_delay = 0
        elif isinstance(schedule_at, Real):
            start_delay = schedule_at - self.beat()
            if start_delay < 0:
                logging.warning("`schedule_at` argument specified a beat in the past; forking immediately.")
                start_delay = 0
        else:  # it's a MetricPhaseTarget
            if not isinstance(schedule_at, MetricPhaseTarget):
                raise ValueError("`schedule_at` must be either a float or a MetricPhaseTarget")
            # get_nearest_matching_beats returns the nearest match below and above, in order of nearness
            # we want the match above, since it's in the future, so we use max
            start_delay = max(*schedule_at.get_nearest_matching_beats(self.beat())) - self.beat()

        def _process(*args, **kwds):
            # set the implicit variable __clock__ in this thread
            threading.current_thread().__clock__ = child
            child.parent_offset += start_delay

            """
            The whole function we are forking is wrapped in a try/except clause, because we want to be able to kill
            it at will. When and if "kill" is called on the clock, its wait_event is set free and it immediately
            raises a ClockKilledError, which exits us from the process. (It's also possible, but unlikely, that
            we will get a DeadClockError, if we were just in the process of calling wait.)
            """
            # Run the function
            process_function(*args, **kwds)

            # Remove this sub-clock from the children list of the forking parent (self is the parent)
            self._children.remove(child)

            child._killed = True

            if done_callback is not None:
                done_callback()

        # self._run_in_pool(_process, args, kwargs)
        def _start_new_clock():
            with child._entering_wait_condition:
                threading.Thread(target=_process, args=args, kwargs=kwargs, daemon=True).start()
                child._entering_wait_condition.wait()

        self.scheduler.schedule_action(
            self.clock_to_scheduler_time(self.beat() + start_delay),
            _start_new_clock,
            child.clock_id,
            {"description": f"Forking of {child}",
             "acting_clock": self,
             "target_beat": self.beat() + start_delay}
        )

    def __repr__(self):
        child_list = "" if len(self._children) == 0 else ", ".join(str(child) for child in self._children)
        return ("Clock('{}')".format(self.name) if self.name is not None else "UNNAMED") + "[" + child_list + "]"