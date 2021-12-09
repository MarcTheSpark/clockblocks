"""
Module defining the central :class:`Clock` class, as well as the :class:`TimeStamp` class, which records the current
beat in every clock at a given moment.
"""

#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  #
#  This file is part of SCAMP (Suite for Computer-Assisted Music in Python)                      #
#  Copyright © 2020 Marc Evanstein <marc@marcevanstein.com>.                                     #
#                                                                                                #
#  This program is free software: you can redistribute it and/or modify it under the terms of    #
#  the GNU General Public License as published by the Free Software Foundation, either version   #
#  3 of the License, or (at your option) any later version.                                      #
#                                                                                                #
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;     #
#  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.     #
#  See the GNU General Public License for more details.                                          #
#                                                                                                #
#  You should have received a copy of the GNU General Public License along with this program.    #
#  If not, see <http://www.gnu.org/licenses/>.                                                   #
#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  #

from .tempo_envelope import MetricPhaseTarget, TempoEnvelope, TempoHistory
from expenvelope._utilities import _get_extrema_and_inflection_points
from collections import namedtuple
from multiprocessing.pool import ThreadPool
from threading import Lock
import logging
from copy import deepcopy
import inspect
from .utilities import sleep_precisely_until, current_clock, _PrintColors
from .debug import _print_and_clear_debug_calc_times
from functools import total_ordering, wraps
import textwrap
from typing import Union, Sequence, Iterator, Tuple, Callable, TypeVar
import time
import threading
from .settings import catching_up_child_clocks_threshold_min, catching_up_child_clocks_threshold_max, \
    running_behind_warning_threshold_long, running_behind_warning_threshold_short
from numbers import Real


_WakeUpCall = namedtuple("WakeUpCall", "t clock")


class ClockblocksError(Exception):
    """Base class for clockblocks errors."""
    pass


class ClockKilledError(ClockblocksError):
    """Exception raised when a clock is killed, allowing us to exit the process forked on it."""
    pass


class DeadClockError(ClockblocksError):
    """Exception raised when a clock is asked to do something after it was killed."""
    pass


class WokenEarlyError(ClockblocksError):
    """Exception raised when master clock is woken up during its wait call."""
    pass


def tempo_modification(fn):
    """
    Decorator applied to methods of Clock that alter tempo. While there's no issue if a clock's tempo is altered
    from its own thread, or even from the thread of a child clock, there would be an issue if it were to
    happen on a parent clock. (This is because the parent clock will have already ceded control to the child,
    and the child will have already scheduled its wakeup call, which may now be changing due to the tempo change.)
    This is also needed when tempo is changed from a non-clock thread.
    """
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        # if we're altering tempo, we should also stop any looping tempo function or envelope currently there
        self._envelope_loop_or_function = None
        if current_clock() == self or (current_clock() is not None
                                       and self in current_clock().iterate_inheritance()):
            fn(self, *args, **kwargs)
        else:
            self.rouse_and_hold()
            fn(self, *args, **kwargs)
            self.release_from_suspension()
    return wrapper


T = TypeVar('T', bound='Clock')


class Clock:
    """
    Recursively nestable clock class. Clocks can fork child-clocks, which can in turn fork their own child-clock.
    Only the master clock calls sleep; child-clocks instead register _WakeUpCalls with their parents, who
    register wake-up calls with their parents all the way up to the master clock.

    :param name: (optional) can be useful for keeping track in confusing multi-threaded situations
    :param parent: the parent clock for this clock; a value of None indicates the master clock
    :param initial_rate: starting rate of this clock (if set, don't set initial tempo or beat length)
    :param initial_tempo: starting tempo of this clock (if set, don't set initial rate or beat length)
    :param initial_beat_length: starting beat length of this clock (if set, don't set initial tempo or rate)
    :param timing_policy: either "relative", "absolute", or a float between 0 and 1 representing a balance between
        the two. "relative" attempts to keeps each wait call as faithful as possible to what it should be. This can
        result in the clock getting behind real time, since if heavy processing causes us to get behind on one note
        we never catch up. "absolute" tries instead to stay faithful to the time since the clock began. If one wait
        is too long due to heavy processing, later delays will be shorter to try to catch up. This can result in
        inaccuracies in relative timing. Setting the timing policy to a float between 0 and 1 implements a hybrid
        approach in which, when the clock gets behind, it is allowed to catch up somewhat, but only to a certain
        extent. (0 is equivalent to absolute timing, 1 is equivalent to relative timing.)
    :param synchronization_policy: either None or one of "all relatives", "all descendants", "no synchronization",
        or "inherit". Since a clock is woken up by its parent clock, it will always remain synchronized with
        all parents / grandparents / etc; however, if you ask one of its children what time / beat it is on, it may
        have old information, since it has been asleep. If the synchronization_policy is set to "no synchronization",
        then we live with this, but if it is set to "all descendants" then we take the time (and CPU cycles) to catch
        up all its descendants so that they read the correct time. Nevertheless, cousin clocks (other descendants of
        this clock's parent) may still not be caught up, so the "all relatives" policy makes sure that all descendants
        of the master clock - no matter how they are related to this clock - will have up-to-date information about
        what time / beat they are on whenever this clock wakes up. This is the default setting, since it avoids
        inaccurate information, but if there are a lot of clocks it may be valuable to turn off relative synchronization
        if it's slowing things down. The value "inherit" means that this clock inherits its synchronization policy from
        its master. If no value is specified, then it defaults to "all relatives" for the master clock and "inherit"
        for all descendants, which in practice means that all clocks will synchronize with all relatives upon waking.
    :param pool_size: the size of the process pool for unsynchronized forks, which are used for playing notes. Only
        has an effect if this is the master clock.
    :ivar name: the name of this clock (string)
    :ivar parent: the parent Clock to which this clock belongs (Clock, or None if master clock)
    :ivar tempo_history: TempoHistory describing how this clock has changed or will change tempo
    """

    def __init__(self, name: str = None, parent: 'Clock' = None, initial_rate: float = None,
                 initial_tempo: float = None, initial_beat_length: float = None,
                 timing_policy: Union[str, float] = 0.98, synchronization_policy: str = None, pool_size: int = 200):

        self.name = name
        self.parent = parent
        if self.parent is not None and self not in self.parent._children:
            self.parent._children.append(self)
        self._children = []

        # queue of WakeUpCalls for child clocks
        self._queue = []
        # used when a tempo change affecting a dormant child clock requires that clock's wakeup call to be removed
        # recalculated, and placed back in the queue. We shouldn't act on the queue until the wake up call is replaced.
        self._queue_lock = Lock()

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

        # how long had my parent been around when I was created
        self.parent_offset = self.parent.beat() if self.parent is not None else 0
        # boolean marking whether this clock is in the middle of a wait call, and event used for that wait call
        self._dormant = False
        self._wait_event = threading.Event()
        # used by child clocks to see if they were woken normally by the parent, or early via an external rouse call
        self._woken_early = False

        self._wait_keeper = _WaitKeeper()

        if self.is_master():
            # The thread pool runs on the master clock
            self._pool = ThreadPool(processes=pool_size)
            # Used to keep track of if we're using all the threads in the pool
            # if so, we just start a thread and throw a warning to increase pool size
            self._pool_semaphore = threading.BoundedSemaphore(pool_size)
            threading.current_thread().__clock__ = self

            # the master clock also holds onto a dictionary of time stamp data
            # (i.e. a dictionary of (time_in_master -> {clock: beat in clock for clock in master.all_descendants}
            self.time_stamp_data = {}
        else:
            # All other clocks just use self.master._pool
            self._pool = None
            self._pool_semaphore = None
            # no need for this unless on the master clock
            self.time_stamp_data = None

        # these are set on the first call to "wait"; this way, any processing at the very beginning is ignored
        self._last_sleep_time = self._start_time = None

        self._timing_policy = timing_policy

        self._synchronization_policy = synchronization_policy if synchronization_policy is not None \
            else "all relatives" if self.is_master() else "inherit"

        self._log_processing_time = False
        self._fast_forward_goal = None

        self._envelope_loop_or_function = None

        self._running_behind_warning_count = 0

        self._killed = False

    ##################################################################################################################
    #                                                  Family Matters
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
    #                                                 Policy Matters
    ##################################################################################################################

    @property
    def synchronization_policy(self) -> str:
        """
        Determines whether or not this clock actively updates the beat and time of related clocks when it wakes up.
        Allowable values are "all relatives", "all descendants", "no synchronization", and "inherit". If "inherit" then
        it will adopt the setting of its parent clock. The default is to keep all clocks up-to-date. Turning off
        synchronization can save calculation time, but it can lead to clocks giving old information about what beat
        or time it is when asked from a different clock's process.
        """
        return self._synchronization_policy

    @synchronization_policy.setter
    def synchronization_policy(self, value):
        if value not in ("all relatives", "all descendants", "no synchronization", "inherit"):
            raise ValueError('Invalid synchronization policy "{}". Must be one of ("all relatives", "all descendants", '
                             '"no synchronization", "inherit").'.format(value))
        if self.is_master() and value == "inherit":
            raise ValueError("Master cannot inherit synchronization policy.")
        self._synchronization_policy = value

    def _resolve_synchronization_policy(self):
        # resolves a value of "inherit" if necessary
        for clock in self.iterate_inheritance():
            if clock.synchronization_policy != "inherit":
                return clock.synchronization_policy

    @property
    def timing_policy(self) -> Union[str, float]:
        """
        Determines how this clock should respond to getting behind.
        Allowable values are "absolute", "relative" or a float between 0 and 1 representing a balance between
        absolute and relative.
        """
        return self._timing_policy

    @timing_policy.setter
    def timing_policy(self, value):
        assert value in ("absolute", "relative") or isinstance(value, (int, float)) and 0 <= value <= 1.
        self._timing_policy = value

    def use_absolute_timing_policy(self) -> None:
        """
        This timing policy only cares about keeping the time since the clock start accurate to what it should be.
        The downside is that relative timings get distorted when it falls behind.
        """
        self._timing_policy = "absolute"

    def use_relative_timing_policy(self) -> None:
        """
        This timing policy only cares about making each individual wait call as accurate as possible.
        The downside is that long periods of calculation cause the clock to drift and get behind.
        """
        self._timing_policy = "relative"

    def use_mixed_timing_policy(self, absolute_relative_mix: float) -> None:
        """
        Balance considerations of relative timing and absolute timing accuracy according to the given coefficient

        :param absolute_relative_mix: a float representing the minimum proportion of the ideal wait time we are willing
            to wait in order to catch up to the correct absolute time since the clock started.
        """
        if not (0.0 <= absolute_relative_mix <= 1.0):
            raise ValueError("Mix coefficient should be between 0 (fully absolute timing policy) "
                             "and 1 (fully relative timing policy).")
        self._timing_policy = absolute_relative_mix

    ##################################################################################################################
    #                                               Timing Properties
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

    def time_in_master(self) -> float:
        """
        How much time (in seconds) has passed since the master clock was created.
        """
        return self.master.time()

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
    @tempo_modification
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
    @tempo_modification
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
    @tempo_modification
    def tempo(self, t):
        self.tempo_history.tempo = t

    def absolute_rate(self) -> float:
        """
        The rate of this clock in beats / (true) second, accounting for the rates of all parent clocks.
        (As opposed to rate, which just gives the rate relative to the parent clock.)
        """
        absolute_rate = self.rate if self.parent is None else (self.rate * self.parent.absolute_rate())
        return absolute_rate

    def absolute_tempo(self) -> float:
        """
        The tempo of this clock in beats / (true) minute, accounting for the rates of all parent clocks.
        (As opposed to tempo, which just gives the tempo relative to the parent clock.)
        """
        return self.absolute_rate() * 60

    def absolute_beat_length(self) -> float:
        """
        The beat length of this clock in (true) seconds, accounting for the rates of all parent clocks.
        (As opposed to beat_length, which just gives the length in parent beats.)
        """
        return 1 / self.absolute_rate()

    ##################################################################################################################
    #                                                  Tempo Changes
    ##################################################################################################################

    @tempo_modification
    def set_beat_length_target(self, beat_length_target: float, duration: float, curve_shape: float = 0,
                               metric_phase_target: Union[float, MetricPhaseTarget, Tuple] = None,
                               duration_units: str = "beats", truncate: bool = True) -> None:
        """
        Set a target beat length for this clock to reach in duration beats/seconds (with the unit defined by
        duration_units).

        :param beat_length_target: The beat length we want to reach
        :param duration: How long until we reach that beat length
        :param curve_shape: > 0 makes change happen later, < 0 makes change happen sooner
        :param metric_phase_target: This argument lets us align the arrival at the given beat length with a particular
            part of the parent beat (time), or, if we specified "time" as our duration units, it allows us to align
            the arrival at that specified time with a particular part of this clock's beat. This argument takes either
            a float in [0, 1), a MetricPhaseTarget object, or a tuple of arguments to the MetricPhaseTarget constructor
        :param duration_units: one of ("beats", "time"); defines whether the duration is in beats or in
            seconds/parent beats.
        :param truncate: Whether or not to delete all future tempo plans before setting this goal.
        """
        self.tempo_history.set_beat_length_target(beat_length_target, duration, curve_shape, metric_phase_target,
                                                  duration_units, truncate)

    @tempo_modification
    def set_rate_target(self, rate_target: float, duration: float, curve_shape: float = 0,
                        metric_phase_target: Union[float, MetricPhaseTarget, Tuple] = None,
                        duration_units: str = "beats", truncate: bool = True) -> None:
        """
        Set a target rate for this clock to reach in duration beats/seconds (with the unit defined by duration_units)

        :param rate_target: The rate we want to reach
        :param duration: How long until we reach that tempo
        :param curve_shape: > 0 makes change happen later, < 0 makes change happen sooner
        :param metric_phase_target: This argument allows us to align the arrival at the given rate with a particular
            part of the parent beat (time), or, if we specified "time" as our duration units, it allows us to align
            the arrival at that specified time with a particular part of this clock's beat. This argument takes either
            a float in [0, 1), a MetricPhaseTarget object, or a tuple of arguments to the MetricPhaseTarget constructor
        :param duration_units: one of ("beats", "time"); defines whether the duration is in beats or in
            seconds/parent beats.
        :param truncate: Whether or not to delete all future tempo plans before setting this goal.
        """
        self.tempo_history.set_rate_target(rate_target, duration, curve_shape, metric_phase_target,
                                           duration_units, truncate)

    @tempo_modification
    def set_tempo_target(self, tempo_target: float, duration: float, curve_shape: float = 0,
                         metric_phase_target: Union[float, MetricPhaseTarget, Tuple] = None,
                         duration_units: str = "beats", truncate: bool = True) -> None:
        """
        Set a target tempo for this clock to reach in duration beats/seconds (with the unit defined by duration_units)

        :param tempo_target: The tempo we want to reach
        :param duration: How long until we reach that tempo
        :param curve_shape: > 0 makes change happen later, < 0 makes change happen sooner
        :param metric_phase_target: This argument allows us to align the arrival at the given tempo with a particular
            part of the parent beat (time), or, if we specified "time" as our duration units, it allows us to align
            the arrival at that specified time with a particular part of this clock's beat. This argument takes either
            a float in [0, 1), a MetricPhaseTarget object, or a tuple of arguments to the MetricPhaseTarget constructor
        :param duration_units: one of ("beats", "time"); defines whether the duration is in beats or in
            seconds/parent beats.
        :param truncate: Whether or not to delete all future tempo plans before setting this goal.
        """
        self.tempo_history.set_tempo_target(tempo_target, duration, curve_shape, metric_phase_target,
                                            duration_units, truncate)

    @tempo_modification
    def set_beat_length_targets(self, beat_length_targets: Sequence[float], durations: Sequence[float],
                                curve_shapes: Sequence[float] = None,
                                metric_phase_targets: Sequence[Union[float, MetricPhaseTarget, Tuple]] = None,
                                duration_units: str = "beats", truncate: bool = True, loop: bool = False) -> None:
        """
        Same as set_beat_length_target, except that you can set multiple targets at once by providing lists to each
        of the arguments.

        :param beat_length_targets: list of the target beat_lengths
        :param durations: list of segment durations (in beats or seconds, as defined by duration_units)
        :param curve_shapes: list of segment curve_shapes (or none to not set curve shape)
        :param metric_phase_targets: list of metric phase targets for each segment (or None to ignore metric phase)
        :param duration_units: one of ("beats", "time"); defines whether the duration is in beats or in
            seconds/parent beats.
        :param truncate: Whether or not to delete all future tempo plans before setting these targets.
        :param loop: If true, loops the added sequence of targets indefinitely, or until stop_tempo_loop_or_function
            is called.
        """
        self.tempo_history.set_beat_length_targets(beat_length_targets, durations, curve_shapes, metric_phase_targets,
                                                   duration_units, truncate)
        if loop:
            self._loop_segments(self.tempo_history.segments[-len(beat_length_targets):])

    def _loop_segments(self, segments_to_loop):
        self._envelope_loop_or_function = TempoEnvelope(
            [s.start_level for s in segments_to_loop] + [segments_to_loop[-1].end_level],
            [s.duration for s in segments_to_loop],
            [s.curve_shape for s in segments_to_loop],
            units="beatlength"
        )

    @tempo_modification
    def set_rate_targets(self, rate_targets: Sequence[float], durations: Sequence[float],
                         curve_shapes: Sequence[float] = None,
                         metric_phase_targets: Sequence[Union[float, MetricPhaseTarget, Tuple]] = None,
                         duration_units: str = "beats", truncate: bool = True, loop: bool = False) -> None:
        """
        Same as set_rate_target, except that you can set multiple targets at once by providing lists to each
        of the arguments.

        :param rate_targets: list of the target rates
        :param durations: list of segment durations (in beats or seconds, as defined by duration_units)
        :param curve_shapes: list of segment curve_shapes (or none to not set curve shape)
        :param metric_phase_targets: list of metric phase targets for each segment (or None to ignore metric phase)
        :param duration_units: one of ("beats", "time"); defines whether the duration is in beats or in
            seconds/parent beats.
        :param truncate: Whether or not to delete all future tempo plans before setting these targets.
        :param loop: If true, loops the added sequence of targets indefinitely, or until stop_tempo_loop_or_function
            is called.
        """
        self.tempo_history.set_rate_targets(rate_targets, durations, curve_shapes, metric_phase_targets,
                                            duration_units, truncate)
        if loop:
            self._loop_segments(self.tempo_history.segments[-len(rate_targets):])

    @tempo_modification
    def set_tempo_targets(self, tempo_targets: Sequence[float], durations: Sequence[float],
                          curve_shapes: Sequence[float] = None,
                          metric_phase_targets: Sequence[Union[float, MetricPhaseTarget, Tuple]] = None,
                          duration_units: str = "beats", truncate: bool = True, loop: bool = False) -> None:
        """
        Same as set_tempo_target, except that you can set multiple targets at once by providing lists to each
        of the arguments.

        :param tempo_targets: list of the target tempos
        :param durations: list of segment durations (in beats or seconds, as defined by duration_units)
        :param curve_shapes: list of segment curve_shapes (or none to not set curve shape)
        :param metric_phase_targets: list of metric phase targets for each segment (or None to ignore metric phase)
        :param duration_units: one of ("beats", "time"); defines whether the duration is in beats or in
            seconds/parent beats.
        :param truncate: Whether or not to delete all future tempo plans before setting these targets.
        :param loop: If true, loops the added sequence of targets indefinitely, or until stop_tempo_loop_or_function
            is called.
        """
        self.tempo_history.set_tempo_targets(tempo_targets, durations, curve_shapes, metric_phase_targets,
                                             duration_units, truncate)
        if loop:
            self._loop_segments(self.tempo_history.segments[-len(tempo_targets):])

    def stop_tempo_loop_or_function(self) -> None:
        """
        If we set up a looping list of tempo targets, this stops that looping process. It also stops extending any
        tempo function that we set up to go indefinitely.
        """
        self._envelope_loop_or_function = None

    # --------------------------------------------- Tempo Functions -------------------------------------------------

    @tempo_modification
    def apply_beat_length_function(self, function: Callable, domain_start: float = 0, domain_end: float = None,
                                   duration_units: str = "beats", truncate: bool = True, loop: bool = False,
                                   extension_increment: float = 1.0, resolution_multiple: int = 2) -> None:
        """
        Applies a function to be used to set the beat_length (and therefore rate and tempo) of this clock.

        :param function: a function (usually a lambda function) taking a single argument representing the beat or the
            time (depending on the 'duration_units' argument) and outputting the beat_length. Note that the beat or
            time argument is measured in relation to the moment that apply_beat_length_function was called, not from
            the start of the clock.
        :param domain_start: along with domain_end, allows us to specify a specific portion of the function to be used
        :param domain_end: along with domain_start, allows us to specify a specific portion of the function to be used.
            If None, the tempo continues to be defined according to the function formula.
        :param duration_units: either "beats" or "time", depending on what the function argument refers to
        :param truncate: whether or not to remove all future plans before applying this function
        :param loop: If a finite portion of the function's domain is being used, this argument will loop that portion
            of the domain when we come to the end of it.
        :param extension_increment: if domain_end is None, then this defines how far in advance we extend the function
            at any given time.
        :param resolution_multiple: determines how precisely the function is to be approximated
        """
        self._apply_tempo_function(function, domain_start=domain_start, domain_end=domain_end, units="beatlength",
                                   duration_units=duration_units, truncate=truncate, loop=loop,
                                   extension_increment=extension_increment, resolution_multiple=resolution_multiple)

    @tempo_modification
    def apply_rate_function(self, function: Callable, domain_start: float = 0, domain_end: float = None,
                            duration_units: str = "beats", truncate: bool = True, loop: bool = False,
                            extension_increment: float = 1.0, resolution_multiple: int = 2) -> None:
        """
        Applies a function to be used to set the rate (and therefore beat_length and tempo) of this clock.

        :param function: a function (usually a lambda function) taking a single argument representing the beat or the
            time (depending on the 'duration_units' argument) and outputting the rate. Note that the beat or
            time argument is measured in relation to the moment that apply_beat_length_function was called, not from
            the start of the clock.
        :param domain_start: along with domain_end, allows us to specify a specific portion of the function to be used
        :param domain_end: along with domain_start, allows us to specify a specific portion of the function to be used.
            If None, the tempo continues to be defined according to the function formula.
        :param duration_units: either "beats" or "time", depending on what the function argument refers to
        :param truncate: whether or not to remove all future plans before applying this function
        :param loop: If a finite portion of the function's domain is being used, this argument will loop that portion
            of the domain when we come to the end of it.
        :param extension_increment: if domain_end is None, then this defines how far in advance we extend the function
            at any given time.
        :param resolution_multiple: determines how precisely the function is to be approximated
        """
        self._apply_tempo_function(function, domain_start=domain_start, domain_end=domain_end, units="rate",
                                   duration_units=duration_units, truncate=truncate, loop=loop,
                                   extension_increment=extension_increment, resolution_multiple=resolution_multiple)

    @tempo_modification
    def apply_tempo_function(self, function: Callable, domain_start: float = 0, domain_end: float = None,
                             duration_units: str = "beats", truncate: bool = True, loop: bool = False,
                             extension_increment: float = 1.0, resolution_multiple: int = 2) -> None:
        """
        Applies a function to be used to set the tempo (and therefore beat_length and rate) of this clock.

        :param function: a function (usually a lambda function) taking a single argument representing the beat or the
            time (depending on the 'duration_units' argument) and outputting the tempo in BPM. Note that the beat or
            time argument is measured in relation to the moment that apply_beat_length_function was called, not from
            the start of the clock.
        :param domain_start: along with domain_end, allows us to specify a specific portion of the function to be used
        :param domain_end: along with domain_start, allows us to specify a specific portion of the function to be used.
            If None, the tempo continues to be defined according to the function formula.
        :param duration_units: either "beats" or "time", depending on what the function argument refers to
        :param truncate: whether or not to remove all future plans before applying this function
        :param loop: If a finite portion of the function's domain is being used, this argument will loop that portion
            of the domain when we come to the end of it.
        :param extension_increment: if domain_end is None, then this defines how far in advance we extend the function
            at any given time.
        :param resolution_multiple: determines how precisely the function is to be approximated
        """
        self._apply_tempo_function(function, domain_start=domain_start, domain_end=domain_end, units="tempo",
                                   duration_units=duration_units, truncate=truncate, loop=loop,
                                   extension_increment=extension_increment, resolution_multiple=resolution_multiple)

    def _apply_tempo_function(self, function, domain_start: float = 0, domain_end: float = None,
                              units: str = "beatlength", duration_units: str = "beats", truncate: bool = False,
                              loop: bool = False, extension_increment: float = 1.0,
                              resolution_multiple: int = 2) -> None:
        # truncate removes any segments that extend into the future
        if truncate:
            self.tempo_history.remove_segments_after(self.beat())

        if domain_end is None:
            self.tempo_history.append_envelope(
                TempoEnvelope.from_function(function, domain_start, domain_start + extension_increment,
                                            units=units, duration_units=duration_units,
                                            resolution_multiple=resolution_multiple))
            # add a note to use this function, starting where we left off, and going by the same extension increment
            # when we get to the end of the envelope
            self._envelope_loop_or_function = (function, domain_start + extension_increment, extension_increment,
                                               units, duration_units, resolution_multiple)
        else:
            envelope = TempoEnvelope.from_function(
                function, domain_start, domain_end, units=units,
                duration_units=duration_units, resolution_multiple=resolution_multiple
            )

            if self.tempo_history.length() == 0:
                # if there's nothing to this clock's tempo envelope yet, we just replace it with the new one
                self.tempo_history = envelope
                if loop:
                    # but if we're looping the same envelope that we just set this clocks tempo_envelope to,
                    # we need to make a copy or we start adding an envelope to itself
                    self._envelope_loop_or_function = deepcopy(envelope)
            else:
                # if we're just appending to an existing envelope, then we don't need to make a deep copy if we loop
                self.tempo_history.append_envelope(envelope)
                if loop:
                    self._envelope_loop_or_function = envelope

    ##################################################################################################################
    #                                                   Forking
    ##################################################################################################################

    def _run_in_pool(self, target: Callable, args, kwargs) -> None:
        if self.master._pool_semaphore.acquire(blocking=False):
            semaphore = self.master._pool_semaphore
            self.master._pool.apply_async(target, args=args, kwds=kwargs, callback=lambda _: semaphore.release())
        else:
            logging.warning("Ran out of threads in the master clock's ThreadPool; small thread creation delays may "
                            "result. You can increase the number of threads in the pool to avoid this.")
            threading.Thread(target=target, args=args, kwargs=kwargs, daemon=True).start()

    def fork(self, process_function: Callable, args: Sequence = (), kwargs: dict = None, name: str = None,
             initial_rate: float = None, initial_tempo: float = None, initial_beat_length: float = None,
             schedule_at: Union[float, MetricPhaseTarget] = None) -> 'Clock':
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
        :return: the clock of the spawned child process
        """
        if not self.alive:
            raise DeadClockError("Cannot call fork from a clock that is no longer running.")

        # If we're calling fork from a non clock thread, or from the thread of a parent clock, we need to
        # rouse this clock, since it will be asleep.
        if current_clock() != self and (current_clock() is None or self not in current_clock().iterate_inheritance()):
            self.rouse_and_hold()
            had_to_rouse = True
        else:
            had_to_rouse = False

        kwargs = {} if kwargs is None else kwargs

        name = (process_function.__name__ if hasattr(process_function, '__name__') else "UNNAMED") \
            if name is None else name

        child = Clock(name, parent=self, initial_rate=initial_rate, initial_tempo=initial_tempo,
                      initial_beat_length=initial_beat_length)

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
            try:
                # set the implicit variable __clock__ in this thread
                threading.current_thread().__clock__ = child
                # make sure we have been given a reasonable number of arguments, and see if
                # the number given suggests an expected first clock argument
                process_function_signature = inspect.signature(process_function)
                num_positional_parameters = len([
                    param for param in process_function_signature.parameters if
                    process_function_signature.parameters[param].default == inspect.Parameter.empty
                ])

                """
                The whole function we are forking is wrapped in a try/except clause, because we want to be able to kill
                it at will. When and if "kill" is called on the clock, its wait_event is set free and it immediately
                raises a ClockKilledError, which exits us from the process. (It's also possible, but unlikely, that
                we will get a DeadClockError, if we were just in the process of calling wait.)
                """
                try:
                    if start_delay > 0:
                        # if there's a start delay, then we start the clock on a negative beat and time
                        # so that both arrive at zero when when the forked process starts
                        child.tempo_history._t = -start_delay
                        # child.tempo_history.segments[0].start_level is the initial beat length, so this
                        # modifies the start beat proportionally to arrive at zero
                        child.tempo_history._beat = -start_delay / child.tempo_history.segments[0].start_level
                        child.parent_offset += start_delay
                        child.wait(start_delay, units="time")
                    if len(args) == num_positional_parameters - 1:
                        # if the process_function we have been takes one more argument than
                        # provided then we pass the clock as the first argument
                        process_function(child, *args, **kwds)
                    else:
                        # otherwise we just pass the arguments as given
                        process_function(*args, **kwds)
                except ClockKilledError:
                    pass
                except DeadClockError:
                    pass

                self._children.remove(child)
                child._killed = True
            except Exception as e:
                logging.exception(e)
        self._run_in_pool(_process, args, kwargs)

        # Allow the new child to run until it hits a wait call. This is quite important; if we don't do this,
        # the master clock may end up not seeing any queued events and go to sleep for a while before this newly
        # forked process sets if first wake up call.
        while child in self._children and not child._dormant:
            # note that sleeping a tiny amount is better than a straight while loop,
            # which slows down the other threads with its greediness
            time.sleep(0.000001)

        if had_to_rouse:
            self.release_from_suspension()
        return child

    def kill(self) -> None:
        """
        Ends the process forked on this clock and all child processes.
        If this is the master clock, since it wasn't forked, this simply kills all child processes.
        """
        self._killed = True
        self._wait_event.set()
        for child in self.children():
            child.kill()
        if self.is_master():
            self._pool.terminate()

    @property
    def alive(self) -> bool:
        """
        Whether or not this clock is still running

        :return: True if running, False if it was killed or its process ended.
        """
        return not self._killed

    def fork_unsynchronized(self, process_function: Callable, args: Sequence = (), kwargs: dict = None) -> None:
        """
        Spawns a parallel process, but as an asynchronous thread, not on a child clock.
        (Still makes use of this clock's ThreadPool, so it's quicker to spawn than creating a new Thread)

        :param process_function: the process to be spawned
        :param args: arguments for that function
        :param kwargs: keyword arguments for that function
        """
        kwargs = {} if kwargs is None else kwargs

        def _process(*args, **kwargs):
            try:
                process_function(*args, **kwargs)
            except Exception as e:
                logging.exception(e)

        self._run_in_pool(_process, args, kwargs)

    def run_as_server(self) -> T:
        """
        Runs this clock on a parallel thread so that it can act as a server. This is the approach that should be taken
        if running clockblocks from an interactive terminal session. Simply type :code:`c = Clock().run_as_server()`

        :return: self
        """
        def run_server():
            threading.current_thread().__clock__ = self
            while True:
                current_clock().wait_forever()

        threading.Thread(target=run_server, daemon=True).start()
        # don't have the thread that called this recognize the Session as its clock anymore
        threading.current_thread().__clock__ = None
        return self

    ##################################################################################################################
    #                                             Waiting (the guts)
    ##################################################################################################################

    def _wait_in_parent(self, dt: float) -> None:
        """
        If master clock, sleeps precisely for dt seconds (with adjustments based on timing policy / fast forwarding)
        Otherwise registers a wake up time with parent clock and pauses execution.
        NB: Should not be called directly, since this doesn't actually advance the tempo clock; instead call
        clock.wait(dt, units)

        :param dt: how many beats to wait on the parent clock
        """
        if self._log_processing_time:
            logging.info("Clock {} processed for {} secs.".format(self.name if self.name is not None else "<unnamed>",
                                                                  time.time() - self._last_sleep_time))

        if self.is_master():
            # this is the master thread that actually sleeps
            # ...unless we're fast-forwarding. Better address that possibility.
            if self._fast_forward_goal is not None:
                if self.time() >= self._fast_forward_goal:
                    # done fast-forwarding
                    self._fast_forward_goal = None
                elif self.time() < self._fast_forward_goal <= self.time() + dt:
                    # the fast forward goal is reached in the middle of this wait call,
                    # so we should redefine dt as the remaining time after the fast-forward goal
                    dt = (self.time() + dt) - self._fast_forward_goal
                    # if using absolute timing, pretend that we started playback earlier by the part that we didn't wait
                    self._start_time -= (self._fast_forward_goal - self.time())
                    self._fast_forward_goal = None
                else:
                    # clearly, self._fast_forward_goal >= self.time() + dt, so we're still fast-forwarding.
                    # keep track of _last_sleep_time, but then return without waiting
                    self._last_sleep_time = time.time()
                    # if we're using absolute timing, we need to pretend that we started playback earlier by dt
                    self._start_time -= dt
                    return

            # a relative timing policy means we stop sleeping dt after we last finished sleeping, not including the
            # processing that happened since we woke up. This makes each wait call as accurate as possible
            stop_sleeping_time_relative = self._last_sleep_time + dt
            # an absolute timing policy means we remain faithful to the amount of time that should have passed since
            # the start of the clock. This eliminates the possibility of drift, but might lead to some inaccurate waits
            stop_sleeping_time_absolute = self._start_time + self.time() + dt

            # if self._timing_policy is a float, that represents a compromise between absolute and relative timing
            # we can wait shorter than expected in order to catch up when we get behind, but only down to a certain
            # percentage of the given wait. E.g. 0.8 means that we are guaranteed to wait at least 80% of the wait time
            stop_sleeping_time = stop_sleeping_time_relative if self._timing_policy == "relative" \
                else stop_sleeping_time_absolute if self._timing_policy == "absolute" \
                else max(self._last_sleep_time + dt * self._timing_policy, stop_sleeping_time_absolute)

            if _print_and_clear_debug_calc_times():
                print("MASTER SCHEDULED WAIT TIME: {}, TOTAL PROCESSING TIME: {}".format(
                    dt, time.time() - self._last_sleep_time))

            # in case processing took so long that we are already significantly past the time we were supposed to stop
            # sleeping, we throw a warning that we're getting behind and don't try to sleep at all. The threshold for
            # what is too far behind is set in the settings, and varies depending on how long the wait call is. We
            # actually give *more* leeway by default if the wait call is very short, because otherwise warnings get
            # thrown when clocks almost coincide.
            running_behind_threshold = running_behind_warning_threshold_long if dt > 0.1 else \
                running_behind_warning_threshold_short if dt <= 0 else \
                (1 - dt / 0.1) * (running_behind_warning_threshold_short - running_behind_warning_threshold_long) + \
                running_behind_warning_threshold_long

            if stop_sleeping_time < time.time() - running_behind_threshold:
                # if we're more than 10 ms behind, throw a warning: this starts to get noticeable
                logging.warning(
                    "Clock is running noticeably behind real time ({} s) on a wait call of {} s; "
                    "probably processing is too heavy.".format(
                        round(time.time() - stop_sleeping_time, 5), round(dt, 5))
                )
                self._running_behind_warning_count += 1
            elif stop_sleeping_time < time.time():
                # we're running a tiny bit behind, but not noticeably, so just don't sleep and let it be what it is
                pass
            else:
                self._dormant = True
                if dt > 1e-5:
                    sleep_precisely_until(stop_sleeping_time, self._wait_event)
                self._dormant = False
                if self._wait_event.is_set():
                    self._wait_event.clear()
                    raise WokenEarlyError()
        else:
            self.parent._queue.append(_WakeUpCall(self.parent.beat() + dt, self))
            self.parent._queue.sort(key=lambda x: x.t)
            if self.parent._queue_lock.locked():
                # if the parent was told to wait while we removed and recalculated the new WakeUpCall, release it now
                self.parent._queue_lock.release()
            self._dormant = True
            self._wait_event.wait()

            if self._woken_early:
                self._woken_early = False
                self._wait_event.clear()
                raise WokenEarlyError()

            if self._killed:
                raise ClockKilledError()
            self._wait_event.clear()
        self._last_sleep_time = time.time()

    def wait(self, dt: float, units: str = "beats", sub_call: bool = False) -> float:
        """
        Causes the current thread to block for dt beats (or seconds if units="time") in this clock.

        :param dt: the number of beats in this clock
        :param units: either "beats" or "time". If time, then it ignores the rate of this clock, and waits in seconds
            (or in parent clock beats, if this is a sub-clock)
        :param sub_call: flag that wouldn't matter to the user, but indicates whether this wait call was called from
            within another wait call.
        :return: 0 if this wait call is uninterrupted, otherwise the amount of beats that remained when interrupted
        """
        if self._start_time is None:
            self._last_sleep_time = self._start_time = time.time()
        units = units.lower()
        if units not in ("beats", "time"):
            raise ValueError("Invalid value of \"{}\" for units. Must be either \"beats\" or \"time\".".format(units))
        if not self.alive:
            raise DeadClockError("Cannot call wait; clock is no longer running.")

        self._wait_for_children_to_finish_processing()

        # make sure any timestamps for this moment have all clocks represented in them
        # (since some new clocks may have been forked since they were created in one of the threads)
        self._complete_timestamp_data()

        end_beat = self._get_wait_end_beat(dt, units)

        try:
            # while there are wake up calls left to do amongst the children, and those wake up calls
            # would take place before or exactly when we're done waiting here on the master clock
            while self._queue_has_wakeup_call(before_beat=end_beat):
                self._handle_next_wakeup_call()

            # if we exit the while loop, that means that there is no one in the queue (meaning no children),
            # or the first wake up call is scheduled for after this wait is to end. So we can safely wait.
            self._wait_in_parent(self.tempo_history.get_wait_time(end_beat - self.beat()))
            beats_passed = end_beat - self.beat()
            woken_early = False
        except WokenEarlyError:
            # clock was roused part-way through the wait call (usually from a thread outside the clock system)
            time_passed = time.time() - self._last_sleep_time
            self._last_sleep_time = time.time()
            beats_passed = self.tempo_history.get_beat_wait_from_time_wait(time_passed)
            if self.parent is not None:
                self.parent._queue_lock.acquire()
                self._remove_wakeup_call_from_parent_queue()

            woken_early = True

        if not woken_early or self.is_master():
            # do this unless we were woken early and are not on the master clock (in that case, the master will be
            # roused and catch up all the children, and our main goal is to recalculate the next wake up call)
            self.tempo_history.advance(beats_passed)
            self._synchronize_children()

        # the wait keeper holds the clock in suspension, generally when the clock has been woken up early (using
        # rouse_and_hold) and we're waiting to carry out a few commands before letting it go do its thing
        self._wait_keeper.wait_for_clearance()

        # if the clock was roused mid-wait, wait the rest of the time
        if woken_early:
            if sub_call:
                return end_beat - self.beat()
            else:
                remainder = end_beat - self.beat()
                while remainder > 0:
                    remainder = self.wait(end_beat - self.beat(), sub_call=True)
        return 0

    def _wait_for_children_to_finish_processing(self) -> None:
        while not all(child._dormant for child in self._children):
            # note that sleeping a tiny amount is better than a straight while loop,
            # which slows down the other threads with its greediness
            time.sleep(0.000001)

    def _synchronize_children(self):
        # see explanation of synchronization_policy in constructor
        start = time.time()
        if self._resolve_synchronization_policy() == "all relatives":
            self.master._catch_up_children()
        elif self._resolve_synchronization_policy() == "all descendants":
            self._catch_up_children()
        calc_time = time.time() - start
        if calc_time > (catching_up_child_clocks_threshold_max if self.master._running_behind_warning_count == 0
                        else catching_up_child_clocks_threshold_min):
            # throw a warning if catching up child clocks is being slow. Be more picky if the master is getting behind
            logging.warning("Catching up child clocks is taking a little while ({} seconds to be precise) on "
                            "clock {}. \nUnless you are recording on a child or cousin clock, you can safely turn this "
                            "off by setting the synchronization_policy for this clock (or for the master clock) to "
                            "\"no synchronization\"".format(calc_time, current_clock().name))
            self.master._running_behind_warning_count = 0

    def _remove_wakeup_call_from_parent_queue(self):
        i = 0
        while i < len(self.parent._queue):
            if self.parent._queue[i].clock == self:
                self.parent._queue.pop(i)
            else:
                i += 1

    def _handle_next_wakeup_call(self):
        # find the next wake up call
        next_wake_up_call = self._queue[0]
        wake_up_beat = next_wake_up_call.t
        beats_till_wake = wake_up_beat - self.beat()
        self._wait_in_parent(self.tempo_history.get_wait_time(beats_till_wake))
        self._queue.pop(0)  # we only pop the wake up call if _wait_in_parent doesn't throw a WokenEarlyException
        self._advance_tempo_map_to_beat(wake_up_beat)
        # tell the process of the clock being woken to go ahead and do it's thing
        next_wake_up_call.clock._dormant = False  # this flag tells us when that process hits a new wait
        next_wake_up_call.clock._wait_event.set()
        self._wait_for_child_to_finish_processing(next_wake_up_call.clock)

    def _wait_for_child_to_finish_processing(self, child_clock: 'Clock') -> None:
        # wait for the child clock that we woke up either to finish completely or to finish processing
        while child_clock in self._children and not child_clock._dormant:
            # note that sleeping a tiny amount is better than a straight while loop,
            # which slows down the other threads with its greediness
            time.sleep(0.000001)

    def _complete_timestamp_data(self):
        # if this is the master clock and a time stamp has been created for this moment
        # make sure to update the data for that time stamp to include all clocks active at that moment
        # this solves the issue of incomplete time stamps that get created just before a new clock is forked
        if self.is_master() and self.time() in self.time_stamp_data:
            for c in self.iterate_all_relatives(include_self=True):
                if c not in self.time_stamp_data[self.time()]:
                    self.time_stamp_data[self.time()][c] = c.beat()

    def _catch_up_children(self):
        # when we catch up the children, they also have to recursively catch up their children, etc.
        for child in self._children:
            if (child.parent_offset + child.time()) < self.beat():
                child.tempo_history.advance_time(self.beat() - (child.parent_offset + child.time()))
                child._catch_up_children()

    def _get_wait_end_beat(self, dt, units):
        end_beat = self.beat() + dt if units == "beats" \
            else self.beat() + self.tempo_history.get_beat_wait_from_time_wait(dt)

        # if we have a looping tempo envelope or an endless tempo function, and we're going right up
        # to or past the end of what's already been charted out, then we extend it before waiting
        extension_needed = self._extend_looping_envelopes_if_needed(end_beat)

        if extension_needed and units == "time":
            # if we're using time units, and we added an extention, then we need to recalculate the end time based on
            # the new information about how the tempo envelope extends
            end_beat = self.beat() + self.tempo_history.get_beat_wait_from_time_wait(dt)

        return end_beat

    def _extend_looping_envelopes_if_needed(self, beat_to_extend_to):
        """
        Call this function if there's a looping envelope or tempo function and we need to build it out more.

        :param beat_to_extend_to: the beat we need to extend the envelope to
        :return: True if an extension was needed, false otherwise
        """
        extension_needed = False

        while beat_to_extend_to >= self.tempo_history.end_time() and self._envelope_loop_or_function is not None:
            extension_needed = True

            if isinstance(self._envelope_loop_or_function, TempoEnvelope):
                self.tempo_history.append_envelope(self._envelope_loop_or_function)
            else:
                function, domain_start, extension_increment, function_units, \
                    function_duration_units, resolution_multiple = self._envelope_loop_or_function

                next_key_point = _get_extrema_and_inflection_points(
                    function, domain_start, domain_start + extension_increment,
                    return_on_first_point=True, iterations=4)

                increment = (next_key_point - domain_start) / resolution_multiple
                for k in range(resolution_multiple):
                    piece_start = domain_start + k * increment
                    piece_end = domain_start + (k + 1) * increment
                    self.tempo_history.append_segment(
                        TempoEnvelope.convert_units(function(piece_end), function_units, "beatlength"), increment,
                        halfway_level=TempoEnvelope.convert_units(function((piece_start + piece_end) / 2),
                                                                  function_units, "beatlength")
                    )
                    if function_duration_units == "time":
                        segment = self.tempo_history.segments[-1]
                        modified_segment_length = segment.duration ** 2 / segment.integrate_segment(segment.start_time,
                                                                                                    segment.end_time)
                        segment.end_time = segment.start_time + modified_segment_length

                # add a note to use this function, starting where we left off, and going by the same extension increment
                # when we get to the end of the envelope
                self._envelope_loop_or_function = (function, next_key_point, extension_increment,
                                                   function_units, function_duration_units, resolution_multiple)

        return extension_needed

    def _advance_tempo_map_to_beat(self, beat):
        self.tempo_history.advance(beat - self.beat())

    def _queue_has_wakeup_call(self, before_beat=None):
        # checks if there's a wakeup call (optional before the given beat)
        # uses the _queue_lock just in case there's a child clock currently recalculating it's wakeup call
        with self._queue_lock:
            return len(self._queue) > 0 and (before_beat is None or self._queue[0].t <= before_beat)

    def wait_for_children_to_finish(self) -> None:
        """
        Causes this thread to block until all forked child processes have finished.
        """
        done = False
        while not done:
            if self._start_time is None:
                self._last_sleep_time = self._start_time = time.time()
            if not self.alive:
                raise DeadClockError("Cannot call wait; clock is no longer live.")

            self._wait_for_children_to_finish_processing()

            # make sure any timestamps for this moment have all clocks represented in them
            self._complete_timestamp_data()

            try:
                # while there are wake up calls left to do amongst the children
                while self._queue_has_wakeup_call():
                    self._handle_next_wakeup_call()
                done = True
            except WokenEarlyError:
                # clock was roused part-way through the wait call (usually from a thread outside the clock system)

                time_passed = time.time() - self._last_sleep_time
                self._last_sleep_time = time.time()
                beats_passed = self.tempo_history.get_beat_wait_from_time_wait(time_passed)

                if self.is_master():
                    # do this unless we were woken early and are not on the master clock (in that case, the master will
                    # be roused and catch up all the children, and our main goal is to recalculate the next wake up call)

                    self.tempo_history.advance(beats_passed)
                    self._synchronize_children()
                else:
                    self.parent._queue_lock.acquire()
                    self._remove_wakeup_call_from_parent_queue()

                # the wait keeper holds the clock in suspension, generally when the clock has been woken up early (using
                # rouse_and_hold) and we're waiting to carry out a few commands before letting it go do its thing
                self._wait_keeper.wait_for_clearance()
                # if woken early, done is False, so we re-enter the loop

    def wait_forever(self) -> None:
        """
        Causes this thread to block forever.
        Generally called from a parent clock when actions are being carried out on child clocks, or as a result of
        callback functions.
        """
        while True:
            self.wait(1.0)

    def rouse_and_hold(self, holding_clock: 'Clock' = None) -> None:
        """
        Rouse this clock if it is dormant, but keep it suspended until release_from_suspension is called.
        Generally this would not be called by the user, but it is useful for callback functions operating outside
        of the clock system, since they can wake up the clock of interest, carry out what they need to carry out
        and then release the clock from suspension.

        :param holding_clock: when rouse_and_hold is called on a child clock, all parents/grandparents/etc. must also
            be roused and held. However, we need to keep track of the child clock that started the whole process. That's
            what this argument is for. (It's an implementation detail).
        """
        if holding_clock is None:
            holding_clock = self

        if not self._dormant:
            # if we're already awake, no need to do anything
            return

        # this will cause this clock to suspend once it reorients, until release_from_suspension is called
        self._wait_keeper.issue_hold(holding_clock)

        if self._dormant:
            self._dormant = False
            self._woken_early = True  # needed in child clocks to know they were woken early
            self._wait_event.set()  # this will trigger a WokenEarlyException in the wait call

        while not self._wait_keeper.being_held:
            time.sleep(0.000001)

        if not self.is_master() and self.parent._dormant:
            self.parent.rouse_and_hold(holding_clock)

    def release_from_suspension(self, releasing_clock: 'Clock' = None) -> None:
        """
        Called after "rouse_and_hold" to release the roused clock.

        :param releasing_clock: the clock providing the initial impetus to release from suspension. Just like
            rouse_and_hold, a release propagates up the family tree, but we need to keep track of the initial clock
            from which it propagates. (Again, this is an implementation detail.
        """
        if releasing_clock is None:
            releasing_clock = self
        self._wait_keeper.release_hold(releasing_clock)
        if not self.is_master():
            self.parent.release_from_suspension(releasing_clock)

    ##################################################################################################################
    #                                                 Fast-forwarding
    ##################################################################################################################

    def fast_forward_to_time(self, t: float) -> None:
        """
        Fast forward clock, skipping instantaneously to the time of t seconds. (Only available on the master clock.)

        :param t: time to fast forward to
        """
        if not self.is_master():
            raise ValueError("Only the master clock can be fast-forwarded.")
        if t < self.time():
            raise ValueError("Cannot fast-forward to a time in the past.")
        self._fast_forward_goal = t

    def fast_forward_in_time(self, t: float) -> None:
        """
        Fast forward clock, skipping ahead instantaneously by t seconds. (Only available on the master clock.)

        :param t: number of seconds to fast forward by
        """
        self.fast_forward_to_time(self.time() + t)

    def fast_forward_to_beat(self, b: float) -> None:
        """
        Fast forward clock, skipping instantaneously to beat b. (Only available on the master clock.)

        :param b: beat to fast forward to
        """
        assert b > self.beat(), "Cannot fast-forward to a beat in the past."
        self.fast_forward_in_beats(b - self.beat())

    def fast_forward_in_beats(self, b: float) -> None:
        """
        Fast forward clock, skipping ahead instantaneously by b beats. (Only available on the master clock.)

        :param b: number of beats to fast forward by
        """
        self.fast_forward_in_time(self.tempo_history.get_wait_time(b))

    def is_fast_forwarding(self) -> bool:
        """
        Determine if the clock is fast-forwarding.

        :return: boolean representing whether the clock is fast-forwarding
        """
        # same as asking if this clock's master clock is fast-forwarding
        return self.master._fast_forward_goal is not None

    ##################################################################################################################
    #                                                 Other Utilities
    ##################################################################################################################

    def log_processing_time(self) -> None:
        """
        Enables logging statements about calculation time between wait calls.
        """
        if logging.getLogger().level > 20:
            logging.warning("Set default logger to level of 20 or less to see INFO logs about clock processing time."
                            " (i.e. call logging.getLogger().setLevel(20))")
        self._log_processing_time = True

    def stop_logging_processing_time(self) -> None:
        """
        Disables logging statements about calculation time between wait calls.
        """
        self._log_processing_time = False

    def _get_time_increments_from_beat_increments(self, increments):
        beat = self.beat()
        time_increments = []
        for increment in increments:
            time_increments.append(
                (self.tempo_history.value_at(beat) + self.tempo_history.value_at(beat + increment)) / 2 * increment
            )
            beat += increment
            self._extend_looping_envelopes_if_needed(beat)
        return time_increments

    def _get_absolute_time_increments_from_beat_increments(self, increments):
        """
        Gets the (absolute) time steps associated with the given beat steps from this point forward.

        Note: was considering using this for unsynchronized fork associated with playback of parameter envelopes in
        an instrument. Maybe I still could

        :param increments: list of beat steps
        :return: list of (absolute) time steps
        """
        clock = self
        while not clock.is_master():
            increments = clock._get_time_increments_from_beat_increments(increments)
            clock = clock.parent
        return clock._get_time_increments_from_beat_increments(increments)

    def extract_absolute_tempo_envelope(self, start_beat: float = 0, step_size: float = 0.1,
                                        tolerance: float = 0.005) -> TempoEnvelope:
        """
        Extracts this clock's absolute TempoHistory (as opposed to the TempoHistory relative to parent clock).
        Used when creating a score from this clock's point of view.

        :param start_beat: where on the TempoHistory to start
        :param step_size: granularity
        :param tolerance: error tolerance with which we allow a step to simply extend the previous segment rather than
            create a new one.
        :return: A TempoHistory representing the true variation of tempo on this clock, as filtered through the
            changing rates of its parents.
        """
        if self.is_master():
            # if this is the master clock, no extraction is necessary; just use its tempo curve
            return self.tempo_history.as_tempo_envelope()

        clocks = self.inheritance()

        tempo_histories = [deepcopy(clock.tempo_history) for clock in clocks]
        tempo_histories[0].go_to_beat(start_beat)
        initial_rate = tempo_histories[0].rate

        for i in range(1, len(tempo_histories)):
            # for each clock, its parent_offset + the time it would take to get to its current beat = its parent's beat
            tempo_histories[i].go_to_beat(clocks[i-1].parent_offset + tempo_histories[i-1].time())
            initial_rate *= tempo_histories[i].rate

        def step_and_get_beat_length(step):
            beat_change = step
            for tempo_history in tempo_histories:
                _, beat_change = tempo_history.advance(beat_change)
            return beat_change / step

        output_curve = TempoEnvelope(initial_rate, units="rate")

        while any(tempo_envelope.beat() < tempo_envelope.length() for tempo_envelope in tempo_histories):
            # we step twice for half the step size so that we can get a halfway point to use to guide curvature
            start_level = output_curve.end_level()
            halfway_level = step_and_get_beat_length(step_size / 2)
            end_level = step_and_get_beat_length(step_size / 2)
            if min(start_level, end_level) < halfway_level < max(start_level, end_level):
                # so long as that halfway point is between the start and end levels, we use it as a guide
                output_curve.append_segment(end_level, step_size, tolerance=tolerance, halfway_level=halfway_level)
            else:
                # but if it's not, then this represents a turn around and we just use two linear segments
                output_curve.append_segment(halfway_level, step_size / 2, tolerance=tolerance)
                output_curve.append_segment(end_level, step_size / 2, tolerance=tolerance)

        return output_curve

    def __repr__(self):
        child_list = "" if len(self._children) == 0 else ", ".join(str(child) for child in self._children)
        return ("Clock('{}')".format(self.name) if self.name is not None else "UNNAMED") + "[" + child_list + "]"


@total_ordering
class TimeStamp:
    """
    A TimeStamp stores the beat of every :class:`Clock` within a family of clocks, at a specific moment.

    :param clock: any clock in the family of clocks we are interested in; if None, the clock is captured implicitly
        from the thread
    :ivar wall_time: the system time (given by time.time()) when this time stamp was created
    :ivar time_in_master: the time in the master clock when this time stamp was created
    :ivar beats_in_clocks: dictionary mapping every clock in the family tree to its beat when the TimeStamp was created.
    """

    def __init__(self, clock: Clock = None):

        clock = current_clock() if clock is None else clock

        if clock is None or not isinstance(clock, Clock):
            raise ValueError("No valid clock given or found for TimeStamp")

        self.wall_time = time.time()
        self.time_in_master = clock.time_in_master() if clock is not None else self.wall_time

        # There's no reason to keep multiple copies of the time stamp data (i.e. beat in each clock) for the
        # same time in the master clock, (also, some of these copies could end up deficient if not all of the
        # child clocks had been forked when they were created). For this reason, we keep all the data in a
        # dictionary in the master clock called time_stamp_data, indexed by the time in master
        if self.time_in_master in clock.master.time_stamp_data:
            # if the given time_in_master is already represented in the time_stamp_data, set self.beats_in_clocks
            # to be an alias to that entry
            self.beats_in_clocks = clock.master.time_stamp_data[self.time_in_master]

            # ...but also update that entry to contain a value for every relative for this clock, in case it was
            # missing one or more
            for c in clock.iterate_all_relatives(include_self=True):
                if c not in self.beats_in_clocks:
                    self.beats_in_clocks[c] = c.beat()
        else:
            # if the given time_in_master is not yet represented in the time_stamp_data, create the entry
            # also, use self.beats_in_clocks as an alias
            self.beats_in_clocks = {
                c: c.beat() for c in clock.iterate_all_relatives(include_self=True)
            }
            clock.master.time_stamp_data[self.time_in_master] = self.beats_in_clocks

    def beat_in_clock(self, clock: Clock) -> float:
        """
        Get what beat it was in the given clock at the moment represented by this TimeStamp.

        :param clock: the clock we are curious about.
        :return: the beat in that clock.
        """
        if clock in self.beats_in_clocks:
            return self.beats_in_clocks[clock]
        raise ValueError("Invalid clock: not found in TimeStamp")

    def time_in_clock(self, clock: Clock) -> float:
        """
        Get what time it was in the given clock at the moment represented by this TimeStamp.

        :param clock: the clock we are curious about.
        :return: the time in that clock.
        """
        if clock.is_master():
            return self.time_in_master
        else:
            return self.beat_in_clock(clock.parent)

    def __repr__(self):
        return "TimeStamp[{}]".format(self.time_in_master)

    def __eq__(self, other):
        return self.time_in_master == other.time_in_master and self.wall_time == other.wall_time

    def __lt__(self, other):
        if self.time_in_master == other.time_in_master:
            return self.wall_time < other.wall_time
        else:
            return self.time_in_master < other.time_in_master


class _WaitKeeper:
    """
    Used by a given clock to allow other clocks to put a hold on it so that it doesn't wait / calculate its next
    wake up call yet. Built so that multiple clocks can issue a hold, and all of them have to release the clock.
    Plays important role in the "rouse_and_hold" process.
    """

    def __init__(self):
        self.blocking_clocks = []
        self._hold_event = threading.Event()
        self._hold_event.set()  # by default, with no holds placed, let's us through
        self.being_held = False

    def issue_hold(self, clock=None):
        """
        Place a hold, issued by the given clock. "wait_for_clearance" will block until it's released.

        :param clock: clock issuing a hold
        """
        clock = current_clock() if clock is None else clock
        assert isinstance(clock, Clock)
        # if clock in self.blocking_clocks:
        #     raise RuntimeError("Cannot issue hold; clock {} has already issued a hold.".format(clock))
        if clock not in self.blocking_clocks:
            self.blocking_clocks.append(clock)
        if len(self.blocking_clocks) > 0:
            self._hold_event.clear()

    def release_hold(self, clock=None):
        """
        Release any a holds issued by the given clock.

        :param clock: clock whose holds to release.
        """
        clock = current_clock() if clock is None else clock
        assert isinstance(clock, Clock)
        # if clock not in self.blocking_clocks:
        #     raise RuntimeError("Cannot release hold; clock {} never placed one.".format(clock))
        if clock in self.blocking_clocks:
            self.blocking_clocks.remove(clock)
        if len(self.blocking_clocks) == 0:
            self._hold_event.set()

    def wait_for_clearance(self):
        """
        Blocks until there are no holds placed by other clocks.
        """
        self.being_held = True
        self._hold_event.wait()
        self.being_held = False
