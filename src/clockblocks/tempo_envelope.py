"""
Module defining the :class:`TempoEnvelope` class for describing a time-varying tempo, the :class:`TempoHistory` class,
which adds to that a tracking of the current beat and time, and the
:class:`~clockblocks.metric_phase.MetricPhaseTarget` class, which specifies a goal arrival point within the
beat (or meter) cycle.
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
import dataclasses
from functools import lru_cache, wraps
from expenvelope import Envelope, EnvelopeSegment
from copy import deepcopy
from clockblocks.utilities import snap_float_to_nice_decimal
from typing import Union, Sequence, Callable
from clockblocks.enums import DurationUnits, TempoUnits
from clockblocks.metric_phase import MetricPhaseTarget


def tempo_modification(fn):
    """
    Decorator applied to methods of TempoEnvelope that change the curve and therefore could mess up any cached
    beat/time relationships.
    """
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        self.time_at_beat.cache_clear()
        self.beat_at_time.cache_clear()
        return fn(self, *args, **kwargs)

    return wrapper


class TempoEnvelope(Envelope):
    r"""
    A subclass of :class:`~expenvelope.envelope.Envelope` that is specifically designed for representing changing tempo
    curves. The underlying envelope represents beat length as a function of the current beat, which means that the
    area under the curve represents how much time should pass from one beat to the next (beats * sec/beat = sec).
    Although the methods take a "units" argument, which can be "beatlength", "tempo", or "rate", these are always
    converted to beat length in the underlying representation

    :param levels: levels of the curve segments (i.e. tempo values) in the units specified by the `units` argument
    :param durations: durations of the curve segments in the units specified by the `duration_units` argument
    :param curve_shapes: see :func:`~expenvelope.envelope.Envelope.from_levels_and_durations`
    :param units: one of "tempo", "rate" or "beat length", determining how we interpret the levels given
    :param duration_units: either "beats" or "time", determining how we interpret the durations given
    """

    def __init__(self, levels: Union[float, Sequence[float]] = (60,), durations: Sequence[float] = (),
                 curve_shapes: Sequence[Union[float, str]] = None,
                 units: str = "tempo", duration_units: str = "beats"):
        units = TempoUnits(units)
        duration_units = DurationUnits(duration_units)

        # Whatever units are given, convert them to the underlying beatlength curve when creating the TempoEnvelope
        super(TempoEnvelope, self).__init__(
            TempoEnvelope.convert_units(levels, units, TempoUnits.BEATLENGTH), durations, curve_shapes, 0
        )

        if duration_units ==  DurationUnits.TIME:
            self.convert_durations_to_times()

    ##################################################################################################################
    #                                                Class Methods
    ##################################################################################################################

    @classmethod
    def from_levels_and_durations(cls, levels: Sequence = (0, 0), durations: Sequence[float] = (0,),
                                  curve_shapes: Sequence[Union[float, str]] = None,
                                  units: str = "tempo", duration_units: str = "beats") -> 'TempoEnvelope':
        """
        Constructs a TempoEnvelope from the given levels, durations and curve shapes, using the specified units.

        :param levels: levels of the curve segments (i.e. tempo values) in the units specified by the `units` argument
        :param durations: durations of the curve segments in the units specified by the `duration_units` argument
        :param curve_shapes: see :func:`~expenvelope.envelope.Envelope.from_levels_and_durations`
        :param units: one of "tempo", "rate" or "beat length", determining how we interpret the levels given
        :param duration_units: either "beats" or "time", determining how we interpret the durations given
        :return: a TempoEnvelope, constructed accordingly
        """
        return cls(levels, durations, curve_shapes, units=units, duration_units=duration_units)

    @classmethod
    def from_levels(cls, levels: Sequence[float], length: float = 1.0, units: str = "tempo",
                    duration_units: str = "beats") -> 'TempoEnvelope':
        """
        Constructs a TempoEnvelope from the given levels and total length, using the specified units.

        :param levels: levels of the curve segments (i.e. tempo values) in the units specified by the `units` argument
        :param length: total length of the tempo curve, in the units specified by the `duration_units` argument
        :param units: one of "tempo", "rate" or "beat length", determining how we interpret the levels given
        :param duration_units: either "beats" or "time", determining how we interpret the durations given
        :return: a TempoEnvelope, constructed accordingly
        """
        return cls(
            *TempoEnvelope._levels_and_length_to_levels_durations_and_curves(levels, length),
            units=units, duration_units=duration_units
        )

    @classmethod
    def from_list(cls, constructor_list: Sequence, units: str = "tempo",
                  duration_units: str = "beats") -> 'TempoEnvelope':
        """
        Construct a TempoEnvelope from a list that can take a number of formats

        :param constructor_list: see :func:`~expenvelope.envelope.Envelope.from_list`
        :param units: one of "tempo", "rate" or "beat length", determining how we interpret the levels given
        :param duration_units: either "beats" or "time", determining how we interpret the durations given
        :return: a TempoEnvelope, constructed accordingly
        """
        assert hasattr(constructor_list, "__len__")
        if hasattr(constructor_list[0], "__len__"):
            # we were given levels and durations, and possibly curvature values
            if len(constructor_list) == 2:
                if hasattr(constructor_list[1], "__len__"):
                    # given levels and durations
                    return cls.from_levels_and_durations(constructor_list[0], constructor_list[1],
                                                         units=units, duration_units=duration_units)
                else:
                    # given levels and the total length
                    return cls.from_levels(constructor_list[0], length=constructor_list[1],
                                           units=units, duration_units=duration_units)

            elif len(constructor_list) >= 3:
                # given levels, durations, and curvature values
                return cls.from_levels_and_durations(constructor_list[0], constructor_list[1], constructor_list[2],
                                                     units=units, duration_units=duration_units)
        else:
            # just given levels
            return cls.from_levels(constructor_list, units=units, duration_units=duration_units)

    @classmethod
    def from_points(cls, *points, units: str = "tempo", duration_units: str = "beats") -> 'TempoEnvelope':
        """
        Construct an envelope from a list of (beat/time, tempo/rate/beat length) pairs. Units are defined by the
        `units` and `duration_units` parameters.

        :param points: list of points, each of which is of the form (time, value) or (time, value, curve_shape)
        :param units: one of "tempo", "rate" or "beat length", determining how we interpret the tempo values
        :param duration_units: either "beats" or "time", determining how we interpret the time values
        :return: a TempoEnvelope, constructed accordingly
        """
        levels, durations, curve_shapes, offset = TempoEnvelope._unwrap_points(*points)
        if offset != 0:
            raise ValueError("TempoEnvelope must start from beat/time zero; when constructing from points, the "
                             "first point must be of the form (0, [start tempo], [optional curve shape]).")
        return cls(levels, durations, curve_shapes, units=units, duration_units=duration_units)

    @classmethod
    def from_function(cls, function, domain_start=0, domain_end=1, units: str = "tempo", duration_units: str = "beats",
                      scanning_step_size: float = 0.05, key_point_resolution_multiple: int = 2, iterations: int = 6,
                      min_key_point_distance: float = 1e-7) -> 'TempoEnvelope':
        """
        Constructs a TempoEnvelope that approximates an arbitrary function. The domain of the function is in units
        defined by the `duration_units` parameter, and the range is in units defined by the `units` parameter.

        :param function: A function from beat/time to tempo/rate/beat length, as defined by the `duration_units` and
            `units` parameters.
        :param domain_start: see :func:`~expenvelope.envelope.Envelope.from_function`
        :param domain_end: see :func:`~expenvelope.envelope.Envelope.from_function`
        :param units: one of "tempo", "rate" or "beat length", determining how we interpret the function output
        :param duration_units: either "beats" or "time", determining how we interpret the function input
        :param scanning_step_size: when analyzing the function for discontinuities, maxima and minima, inflection
            points, etc., use this step size for the initial pass.
        :param key_point_resolution_multiple: factor by which we add extra key points between the extrema and
            inflection points to improve the curve fit.
        :param iterations: when a potential key point is found, we zoom in and scan again in the viscinity of the point.
            This determines how many iterations of zooming we do.
        :param min_key_point_distance: after scanning for key points, any that are closer than this distance are merged.
        :return: a TempoEnvelope, constructed accordingly
        """
        units = TempoUnits(units)
        duration_units = DurationUnits(duration_units)
        converted_function = (lambda x: TempoEnvelope.convert_units(function(x), units, TempoUnits.BEATLENGTH)) \
            if units != TempoUnits.BEATLENGTH else function
        out_envelope = super().from_function(converted_function, domain_start, domain_end,
                                             scanning_step_size=scanning_step_size,
                                             key_point_resolution_multiple=key_point_resolution_multiple,
                                             iterations=iterations, min_key_point_distance=min_key_point_distance)
        if duration_units == DurationUnits.TIME:
            return out_envelope.convert_durations_to_times()
        else:
            return out_envelope

    ##################################################################################################################
    #                                               Basic Functionality
    ##################################################################################################################

    def beat_length_at(self, beat: float, from_left: bool = False) -> float:
        """
        Get the beat length at the given beat. If the beat length jumps at the given beat, the default is to return the
        beat length after the jump, though this can be overridden with the `from_left` argument.

        :param beat: the beat at which to get the beat length
        :param from_left: whether to evaluate from the right or left-hand side of the beat in question
        """
        return self.value_at(beat, from_left)

    def rate_at(self, beat: float, from_left: bool = False) -> float:
        """
        Get the beat rate (in beats/second) at the given beat. If the rate jumps at the given beat, the default is to
        return the rate after the jump, though this can be overridden with the `from_left` argument.

        :param beat: the beat at which to get the rate
        :param from_left: whether to evaluate from the right or left-hand side of the beat in question
        """
        return 1 / self.beat_length_at(beat, from_left)

    def tempo_at(self, beat: float, from_left: bool = False) -> float:
        """
        Get the tempo (in beats/minute) at the given beat. If the tempo jumps at the given beat, the default is to
        return the tempo after the jump, though this can be overridden with the `from_left` argument.

        :param beat: the beat at which to get the tempo
        :param from_left: whether to evaluate from the right or left-hand side of the beat in question
        """
        return self.rate_at(beat, from_left) * 60

    def extend_to(self, beat: float) -> 'TempoEnvelope':
        """
        Extends the end of this TempoEnvelope to the given beat (if needed) by adding a constant segment at the end.
        """
        if self.length() < beat:
            # no explicit segments have been made for a while, insert a constant segment to bring us up to date
            self.append_segment(self.end_level(), beat - self.length())
        return self

    def truncate_at(self, beat: float) -> 'TempoEnvelope':
        """
        Removes all segments after the given beat and adds a constant segment if necessary to bring us up to that beat.

        :param beat: the beat that we are truncating the tempo envelope after
        :return: self, for chaining purposes
        """
        self.remove_segments_after(beat)
        self.extend_to(beat)
        return self

    ##################################################################################################################
    #                                             Conversion Utilities
    ##################################################################################################################

    @staticmethod
    def convert_units(values: Union[float, Sequence[float]], input_units: str,
                      output_units: str) -> Union[float, Sequence[float]]:
        """
        Utility method to convert values between unites of tempo, rate and beat length.

        :param values: value or list of values in terms of the input_units
        :param input_units: current units of the given values (either "tempo", "rate", or "beat length")
        :param output_units: desired units to convert to (either "tempo", "rate", or "beat length")
        :return: the list of values, converted to output units
        """
        input_units = TempoUnits(input_units)
        output_units = TempoUnits(output_units)

        if input_units == output_units:
            return values
        else:
            convert_input_to_beat_length = (lambda x: 1 / x) if input_units == TempoUnits.RATE \
                else (lambda x: 60 / x) if input_units == TempoUnits.TEMPO else (lambda x: x)
            convert_beat_length_to_output = (lambda x: 1 / x) if output_units == TempoUnits.RATE \
                else (lambda x: 60 / x) if output_units == TempoUnits.TEMPO else (lambda x: x)
            if hasattr(values, "__len__"):
                return tuple(convert_beat_length_to_output(convert_input_to_beat_length(x)) for x in values)
            else:
                return convert_beat_length_to_output(convert_input_to_beat_length(values))

    def convert_durations_to_times(self):
        """
        Warps this tempo_curve so that all the locations of key points get re-interpreted as times instead of beat
        locations. For instance, a tempo curve where the rate hovers around 2 will see a segment of length 3 get
        stretched into a segment of length 6, since if it's supposed to take 3 seconds, it would take 6 beats.
        Pretty confusing, but when we want to construct a tempo curve specifying the *times* that changes occur rather
        than the beats, we can first construct it as though the durations were in beats, then call this function
        to warp it so that the durations are in time.

        :return: self, altered accordingly
        """

        # this is a little confusing, like everything else about this function, but it represents the start beat of
        # the curve, which gets scaled inversely to the initial beat length, since a long initial beat length means
        # it won't take that many beats to get to the desired time.
        t = self.start_time() / self.start_level()
        for segment in self.segments:
            """
            The following has been condensed into a single statement for efficiency:

            actual_time_duration = segment.integrate_segment(segment.start_time, segment.end_time)
            # the segment duration is currently in beats, but it also represents the duration we would like
            # the segment to have in time. so the scale factor is desired time duration / actual time duration
            scale_factor = segment.duration / actual_time_duration
            # here's we're scaling the duration to now last as long in time as it used to in beats
            modified_segment_length = scale_factor * segment.duration
            """
            modified_segment_length = segment.duration ** 2 / segment.integrate_segment(segment.start_time,
                                                                                        segment.end_time)
            segment.start_time = t
            t = segment.end_time = t + modified_segment_length
        return self

    ##################################################################################################################
    #                                                Other Utilities
    ##################################################################################################################

    def show_plot(self, title=None, resolution=25, show_segment_divisions=True, units="tempo",
                  x_range=None, y_range=None):
        """
        Shows a plot of this TempoEnvelope using matplotlib.

        :param title: A title to give the plot.
        :param resolution: number of points to use per envelope segment
        :param show_segment_divisions: Whether to place dots at the division points between envelope segments
        :param units: one of "tempo", "rate" or "beat length", determining the units of the y-axis
        :param x_range: min and max value shown on the x-axis
        :param y_range: min and max value shown on the y-axis
        """
        self._construct_plot(title, resolution, show_segment_divisions, units, x_range, y_range).show()

    def _construct_plot(self, title=None, resolution=25, show_segment_divisions=True, units="tempo",
                        x_range=None, y_range=None):
        """
        Constructs and returns (but doesn't show) a plot of this TempoEnvelope.
        """
        # if we're past the end of the envelope, we want to plot that as a final constant segment
        # bring_up_to_date adds that segment, but we don't want to modify the original envelope, so we deepcopy
        if x_range is None:
            x_range = self.start_time(), self.end_time()
        env_to_plot = deepcopy(self).extend_to(x_range[1]) if self.end_time() < x_range[1] else self

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Could not find matplotlib, which is needed for plotting.")

        fig, ax = plt.subplots()
        x_values, y_values = env_to_plot._get_graphable_point_pairs(resolution)
        ax.plot(x_values, TempoEnvelope.convert_units(y_values, TempoUnits.BEATLENGTH, units))
        if show_segment_divisions:
            ax.plot(env_to_plot.times, TempoEnvelope.convert_units(env_to_plot.levels, TempoUnits.BEATLENGTH, units), 'o')
        plt.xlabel("Beat")
        plt.ylabel("Tempo" if units == TempoUnits.TEMPO else "Rate" if units == TempoUnits.RATE else "Beat Length")
        plt.xlim(x_range)
        if y_range is not None:
            plt.ylim(y_range)
        ax.set_title('Graph of TempoEnvelope' if title is None else title)
        return plt

    @classmethod
    def _from_dict(cls, json_dict):
        curve_shapes = None if 'curve_shapes' not in json_dict else json_dict['curve_shapes']
        if 'length' in json_dict:
            return cls.from_levels(json_dict['levels'], json_dict['length'])
        else:
            return cls.from_levels_and_durations(json_dict['levels'], json_dict['durations'],
                                                 curve_shapes)

    def is_default(self) -> bool:
        """
        True if this is the trivial default tempo — a constant 60 bpm (rate 1.0) with no duration, i.e.
        structurally identical to a freshly-constructed ``TempoEnvelope()``. (Note this is stricter than
        "functionally constant at 60": a flat-60 envelope that carries a real duration or extra segments is
        *not* default (and worth surfacing for debugging)
        """
        return self._to_dict() == {'levels': (1.0, 1.0), 'length': 0}

    def __repr__(self):
        return "TempoEnvelope({}, {}, {})".format(
            TempoEnvelope.convert_units(self.levels, TempoUnits.BEATLENGTH, TempoUnits.TEMPO),
            self.get_durations(rounded=True), self.curve_shapes)


@dataclasses.dataclass
class FunctionFollowInfo:
    """
    Internal bookkeeping for a tempo curve that follows a user function: the function itself plus the
    state needed to keep extending the envelope to approximate it as the clock advances.
    """
    func: Callable[[float], float]
    next_domain_start: float
    extension_increment: float
    units: TempoUnits
    duration_units: DurationUnits
    scanning_step_size: float
    min_key_point_distance: float
    iterations: int
    key_point_resolution_multiple: int
    current_end_beat: float
    current_end_time: float


@dataclasses.dataclass
class EnvelopeLoopInfo:
    """
    Internal bookkeeping for a tempo envelope set to loop: the envelope being repeated plus the
    running beat/time extents needed to keep appending copies as the clock advances.
    """
    env: TempoEnvelope
    length_in_beats: float
    length_in_time: float
    current_end_beat: float
    current_end_time: float


class TempoHistory(TempoEnvelope):
    """
    Subclass of TempoEnvelope that keeps track of a current beat and time, and provides functionality for moving
    forward a certain number of beats or seconds, and/or setting tempo target(s) to reach in the future.

    :param levels: see :class:`TempoEnvelope`
    :param durations: see :class:`TempoEnvelope`
    :param curve_shapes: see :class:`TempoEnvelope`
    :param units: see :class:`TempoEnvelope`
    :param duration_units: see :class:`TempoEnvelope`
    :param beat: Where to set the current beat
    """

    def __init__(self, levels: Union[float, Sequence[float]] = (60,), durations: Sequence[float] = (),
                 curve_shapes: Sequence[Union[float, str]] = None,
                 units: str = "tempo", duration_units: str = "beats", beat: float = 0.0):
        super().__init__(levels, durations, curve_shapes, units, duration_units)
        self._t = self._beat = None
        self.go_to_beat(beat)  # this sets _beat and _t
        self.follow_func_or_envelope_loop: Union[FunctionFollowInfo, EnvelopeLoopInfo] = None

    @classmethod
    def from_tempo_envelope(cls, tempo_envelope: TempoEnvelope, beat: float = 0.0):
        """
        Constructs a TempoHistory from a TempoEnvelope.

        :param tempo_envelope: the TempoEnvelope to copy
        :param beat: the beat to start this TempoHistory on
        """
        return cls(tempo_envelope.levels, tempo_envelope.durations, tempo_envelope.curve_shapes,
                   units=TempoUnits.BEATLENGTH, beat=beat)

    ##################################################################################################################
    #                                                 Basic Properties
    ##################################################################################################################

    @property
    def time(self):
        """
        The current time. Time is found by integrating under the beat length curve: seconds/beat * beats = seconds.
        """
        return self._t

    @property
    def beat(self):
        """
        The current beat.
        """
        return self._beat

    @property
    def beat_length(self):
        """
        The current beat length.
        """
        return self.beat_length_at(self._beat)

    @beat_length.setter
    @tempo_modification
    def beat_length(self, beat_length):
        self.truncate()
        if len(self.segments) == 1 and self.length() == 0:
            # if this is an essentially empty tempo envelope, reset its starting beat_length to the given value
            self.segments[0].start_level = self.segments[0].end_level = beat_length
        else:
            self.append_segment(beat_length, 0)

    @property
    def rate(self):
        """
        The current rate in beats/second
        """
        return 1 / self.beat_length

    @rate.setter
    @tempo_modification
    def rate(self, rate):
        self.beat_length = 1 / rate

    @property
    def tempo(self):
        """
        The current tempo in beats/minute
        """
        return self.rate * 60

    @tempo.setter
    @tempo_modification
    def tempo(self, tempo):
        self.rate = tempo / 60

    @lru_cache(maxsize=32)
    def time_at_beat(self, beat):
        """
        Determine the time at the given beat.

        For beats at or ahead of the committed pointer (the common case, used e.g. by wait()), we
        integrate forward from the pointer (cheap). For beats behind the pointer (e.g. when resolving
        TimeStamps) we integrate from the origin instead to avoid a negative integral.

        :param beat: The beat at which to calculate the time.
        """
        if self.follow_func_or_envelope_loop is not None:
            # extend when at or past the current end beat
            # we use <= instead of strict < because otherwise it misses a jump discontinuity
            while self.follow_func_or_envelope_loop.current_end_beat <= beat:
                self._extend_function_or_envelope_loop()
        if beat >= self._beat:
            time_at_beat = self._t + self.integrate_interval(self._beat, beat)
        else:
            time_at_beat = self.integrate_interval(0, beat)
        return snap_float_to_nice_decimal(time_at_beat)

    @lru_cache(maxsize=32)
    def beat_at_time(self, t):
        """
        Determine the beat at the given time. See :meth:`time_at_beat` for why we branch on whether
        the target is ahead of or behind the committed pointer.

        :param t: The time at which to calculate the beat
        """
        if self.follow_func_or_envelope_loop is not None:
            while self.follow_func_or_envelope_loop.current_end_time <= t:
                self._extend_function_or_envelope_loop()
        if t >= self._t:
            beat_at_time = self.get_upper_integration_bound(self._beat, t - self._t, max_error=1e-14)
        else:
            beat_at_time = self.get_upper_integration_bound(0, t, max_error=1e-14)
        return snap_float_to_nice_decimal(beat_at_time)

    ##################################################################################################################
    #                                                Tempo Changes
    ##################################################################################################################

    @tempo_modification
    def append_envelope(self, envelope_to_append: TempoEnvelope, truncate: bool = False,
                        loop: bool = False) -> TempoEnvelope:
        """
        Append another tempo envelope onto the end of this one, starting from the current beat.

        :param envelope_to_append: the TempoEnvelope to add on
        :param truncate: if True, first discard any existing segments that extend past the current beat,
            so the appended envelope begins cleanly from now
        :param loop: if True, keep repeating `envelope_to_append` indefinitely from this point on
        :return: self, for chaining purposes
        """
        # truncate removes any segments that extend into the future
        if truncate:
            self.remove_segments_after(self.beat)
        # add a flat segment up to the current beat if needed
        self.extend_to(self.beat)

        super().append_envelope(envelope_to_append)
        if loop:
            self.follow_func_or_envelope_loop = EnvelopeLoopInfo(
                envelope_to_append,
                envelope_to_append.length(),
                envelope_to_append.integrate_interval(envelope_to_append.start_time(), envelope_to_append.end_time()),
                self.end_time(),
                self.beat + self.integrate_interval(self.beat, self.end_time())
            )
        return self

    @tempo_modification
    def set_beat_length_target(self, beat_length_target: float, duration: float, curve_shape: float = 0,
                               alignment_target: Union[float, 'MetricPhaseTarget', None] = None,
                               duration_units: str = "beats", truncate: bool = True) -> None:
        """
        Set a target beat length for this TempoEnvelope to reach in duration beats/seconds (with the unit defined by
        duration_units).

        :param beat_length_target: The beat length we want to reach
        :param duration: How long until we reach that beat length
        :param curve_shape: > 0 makes change happen later, < 0 makes change happen sooner. When `alignment_target` is
            given, this acts only as a *seed* for solving the curvature, a suggestion rather than a guaranteed
            value. (And even then it only makes sense if alignment target is a MetricPhaseTarget; if it's a fixed
            time/beat, there's only one curvature solution, if it exists.)
        :param alignment_target: optional constraint on the endpoint's *free* axis (the one `duration` does not pin:
            time when duration_units is beats, beats when it is time). Either a number (land the free axis exactly on
            that coordinate, which fully determines the curvature) or a
            :class:`~clockblocks.metric_phase.MetricPhaseTarget` (snap to the nearest matching phase on the free
            axis). See :meth:`_add_segment`. Raises ValueError if unreachable due to
            the limited flexibility of curvature adjustment.
        :param duration_units: one of ("beats", "time"); defines whether the duration is in beats or in seconds.
        :param truncate: Whether or not to truncate this TempoEnvelope to the current beat before setting this target.
        """
        # snapshot so a failed alignment (raising ValueError) leaves the curve untouched
        backup = deepcopy(self.segments)
        try:
            # truncate removes any segments that extend into the future
            if truncate:
                self.remove_segments_after(self.beat)
            # add a flat segment up to the current beat if needed
            self.extend_to(self.beat)
            self._add_segment(beat_length_target, duration, curve_shape, alignment_target, duration_units)
        except Exception:
            self.segments = backup
            raise

    def _add_segment(self, beat_length_target: float, duration: float, curve_shape: float = 0,
                     alignment_target: Union[float, 'MetricPhaseTarget', None] = None,
                     duration_units: str = "beats") -> None:
        """
        The guts of adding a new segment, minus argument checking and truncating/bringing up to date.

        We set a desired beat length target for `duration` `duration_units` (beats/seconds)  in the future.
        We can optionally also give a desired curve shape, and/or `alignment_target`, which constrains the
        endpoint's *free* axis (time when duration_units is beats, beats when it is time). `alignment_target`
        is either a number (land the free axis exactly on that coordinate) or a
        :class:`~clockblocks.metric_phase.MetricPhaseTarget` (snap to the nearest matching phase on the free axis;
        its own `units`, if given, is ignored — the axis
        is already determined from context). Curvature is solved to satisfy it; raises ValueError if no
        candidate is reachable.
        """
        duration_units = DurationUnits(duration_units)
        if duration_units == DurationUnits.BEATS:
            # how far the TempoEnvelope has planned things out already past the current beat
            extension_into_future = self.length() - self.beat
            if duration < extension_into_future:
                raise ValueError("Duration to target must extend beyond the last existing target.")
            self.append_segment(beat_length_target, duration - extension_into_future, curve_shape)
            if alignment_target is not None:
                # since duration_units == BEATS we are setting end beat directly. The free axis is therefore TIME.
                # Bend curvature to land one of the end times indicated by alignment_target
                segment = self.segments[-1]
                if isinstance(alignment_target, MetricPhaseTarget):
                    # A phase target offers a set of matching times; pick the candidates nearest to the segment's
                    # current end time (so any curve_shape given acts as a seed for which phase we aim at).
                    provisional_end_time = self.time + self.integrate_interval(self.beat, segment.end_time)
                    candidates = alignment_target.get_nearest_matching_times(provisional_end_time)
                else:
                    # A fixed target is the single exact end time to hit
                    candidates = (alignment_target,)
                # _solve_segment_end_time either adjusts the end time in place or raises
                if not self._solve_segment_end_time(segment, candidates):
                    raise ValueError(f"Could not bend the curve to align the segment end (in time) to "
                                     f"{alignment_target}.")
        else:
            # duration_units == TIME, so first figure out how far this TempoEnvelope is *already* extended
            # into the future in TIME, and make sure that end time of our desired segment is at least
            # that far in the future
            time_extension_into_future = self.integrate_interval(self.beat, self.length())
            if duration < time_extension_into_future:
                raise ValueError("Duration to target must extend beyond the last existing target.")

            # figure out how long the segment should be (how far it goes past the current TempoCurve end time)
            desired_segment_dur_in_time = duration - time_extension_into_future
            # figure out how long the curve would take if it were only one beat long
            normalized_time = EnvelopeSegment(
                0, 1, self.value_at(self.length()), beat_length_target, curve_shape
            ).integrate_segment(0, 1)
            # Then append the new segment scaling by a factor of desired_segment_dur_in_time / normalized_time
            self.append_segment(beat_length_target, desired_segment_dur_in_time / normalized_time, curve_shape)
            if alignment_target is not None:
                # since duration_units == TIME, the free axis is therefore BEATS.
                # so if an alignment_target is set, we need to bend curvature to land the end *beat*
                segment = self.segments[-1]
                if isinstance(alignment_target, MetricPhaseTarget):
                    # A phase target offers a set of matching beats; pick the candidates nearest the segment's
                    # current end beat (segment.end_time is really the end *beat* in this internal naming).
                    # Note that here also any curvature given acts as a seed, since it affected normalized_time
                    # and therefore the beat length of the segment (desired_segment_dur_in_time / normalized_time)
                    candidates = alignment_target.get_nearest_matching_beats(segment.end_time)
                else:
                    # A fixed target is the single exact end beat to hit
                    candidates = (alignment_target,)
                if not self._solve_segment_end_beat(segment, candidates):
                    raise ValueError(f"Could not bend the curve to align the segment end (in beats) to "
                                     f"{alignment_target}.")

    @tempo_modification
    def set_rate_target(self, rate_target: float, duration: float, curve_shape: float = 0,
                        alignment_target: Union[float, 'MetricPhaseTarget', None] = None,
                        duration_units: str = "beats", truncate: bool = True) -> None:
        """
        Set a target beat rate for this TempoEnvelope to reach in duration beats/seconds (with the unit defined by
        duration_units). See :meth:`set_beat_length_target`.
        """
        self.set_beat_length_target(1 / rate_target, duration, curve_shape, alignment_target,
                                    duration_units, truncate)

    @tempo_modification
    def set_tempo_target(self, tempo_target: float, duration: float, curve_shape: float = 0,
                         alignment_target: Union[float, 'MetricPhaseTarget', None] = None,
                         duration_units: str = "beats", truncate: bool = True) -> None:
        """
        Set a target tempo for this TempoEnvelope to reach in duration beats/seconds (with the unit defined by
        duration_units). See :meth:`set_beat_length_target`.
        """
        self.set_beat_length_target(60 / tempo_target, duration, curve_shape, alignment_target,
                                    duration_units, truncate)

    # -------------------------------------- Axis coordination adjustments -------------------------------------------

    # These two methods are for just adjusting a single segment's metric phase in beat or time.
    # They are used when adding single segments that we want to adjust the phase of

    def _solve_segment_end_time(self, segment, candidate_end_times) -> bool:
        """Holding the segment's end *beat* fixed, bend its curvature so the segment ends at one of
        `candidate_end_times` (tried in order). Returns True on the first reachable candidate (segment
        mutated); else leaves the segment unchanged and returns False. (``set_curvature_to_desired_integral``
        range-checks before mutating, so a failure leaves the segment clean.)"""
        # NB segment.start_time / end_time are really the start and end *beats*.
        segment_start_time = self.time + self.integrate_interval(self.beat, segment.start_time)
        for new_end_time in candidate_end_times:
            try:
                segment.set_curvature_to_desired_integral(new_end_time - segment_start_time)
                return True
            except ValueError:
                pass
        return False

    @staticmethod
    def _solve_segment_end_beat(segment, candidate_end_beats) -> bool:
        """Holding the segment's *time* (integral) fixed, move its end *beat* onto one of
        `candidate_end_beats` (tried in order) and then re-solve curvature to preserve that time. Returns True
        on the first reachable candidate (segment mutated); else restores the segment and returns False.
        (We have to actively restore the segment here, as opposed to in _solve_segment_end_time, because
        each attempt fixes the end beat, mutating the segment, *before* trying to solve the integral.)"""
        original_end_beat = segment.end_time   # 'end_time' is really the end beat
        original_integral = segment.integrate_segment(segment.start_time, segment.end_time)
        for new_end_beat in candidate_end_beats:
            try:
                segment.end_time = new_end_beat
                segment.set_curvature_to_desired_integral(original_integral)
                return True
            except ValueError:
                segment.end_time = original_end_beat   # set_curvature failed after we moved the beat
        return False

    @staticmethod
    def _adjust_segments_time_duration(which_segments: Sequence[EnvelopeSegment], goal_total_time: float) -> bool:
        """
        Adjusts the total time that the segments take without changing the total beats

        :param which_segments: which segments to adjust.
        :param goal_total_time: the total time we want them to take
        :return: True if it's possible (segments mutated), False if not (segments untouched)
        """
        # ranges of how long each segment could take by adjusting curvature
        segment_time_ranges = [segment.get_integral_range() for segment in which_segments]
        # range of how long the entire thing could take
        total_time_range = (sum(x[0] for x in segment_time_ranges), sum(x[1] for x in segment_time_ranges))

        # check if it's even possible to get to the desired time by simply adjusting curvatures
        if not total_time_range[0] < goal_total_time < total_time_range[1]:
            # if not return False
            return False

        # how long each segment currently takes
        segment_times = [segment.integrate_segment(segment.start_time, segment.end_time)
                         for segment in which_segments]
        # how long all the segments take
        total_time = sum(segment_times)

        # on the off-chance that it already works perfectly, there's nothing to adjust — still a success
        if goal_total_time == total_time:
            return True

        # if we've reached this point, we're ready to make the adjustments
        # delta_time is how much of an adjustment we need total
        delta_time = goal_total_time - total_time
        # we distribute this total adjustment between the segments based on how much room they have to move
        # in the direction we want them to move. Longer segments and segments with more room to wiggle do the
        # majority of the adjusting.
        if delta_time < 0:
            weightings = [segment_time - segment_time_range[0]
                          for segment_time, segment_time_range in zip(segment_times, segment_time_ranges)]
        else:
            weightings = [segment_time_range[1] - segment_time
                          for segment_time, segment_time_range in zip(segment_times, segment_time_ranges)]
        weightings_sum = sum(weightings)
        segment_adjustments = [weighting / weightings_sum * delta_time for weighting in weightings]
        for segment, segment_time, segment_adjustment in zip(which_segments, segment_times, segment_adjustments):
            segment.set_curvature_to_desired_integral(segment_time + segment_adjustment)
        return True

    @tempo_modification
    def _align_run(self, run_segments: Sequence[EnvelopeSegment],
                   alignment_target: Union[float, 'MetricPhaseTarget'], free_axis: DurationUnits) -> bool:
        """
        Collectively bend a run of already-appended, contiguous segments (the tail of ``self.segments``)
        so that the run's endpoint lands on ``alignment_target``, a fixed coordinate or MetricPhase target on
        the run's free axis. (The timing of the segments on other, fixed axis will remain unchanged.)

        This is a generalization of :meth:`_solve_segment_end_time` / :meth:`_solve_segment_end_beat` for
        groups of segments: where those methods bend a single segment's curvature, this distributes the bend
        across the whole run. Note that this is underdetermined for a run of length > 1, so we use the
        existing curve shapes of each segement tossed the distribution.

        Returns True on success (the run's segments are mutated in place) and False if no candidate is
        reachable (leaving ``run_segments`` unchanged). Any wider rollback is the caller's responsibility.
        NB ``EnvelopeSegment.start_time`` / ``end_time`` are really start/end *beats* in this internal naming.
        """
        run_start_beat = run_segments[0].start_time
        run_start_time = self.time + self.integrate_interval(self.beat, run_start_beat)

        if free_axis == DurationUnits.TIME:
            # The run pins beats, so the end beat is fixed; push the run's total *time* onto the target by
            # redistributing curvature (beats unchanged). _adjust_segments_time_duration range-checks before
            # mutating, so a failed candidate leaves the run clean and we can try the next.
            run_end_time = run_start_time + sum(seg.integrate_segment(seg.start_time, seg.end_time)
                                                for seg in run_segments)
            if isinstance(alignment_target, MetricPhaseTarget):
                candidate_end_times = alignment_target.get_nearest_matching_times(run_end_time)
            else:
                candidate_end_times = (alignment_target,)
            for goal_end_time in candidate_end_times:
                if self._adjust_segments_time_duration(run_segments, goal_end_time - run_start_time):
                    return True
            return False
        else:
            # The run pins time, so the run's total time is fixed; the free axis is beats. For each candidate
            # end beat we proportionally stretch/squeeze the run's beats to land on it, then re-solve curvature
            # to restore the run's original total time. On failure we restore the run and try the next candidate.
            run_end_beat = run_segments[-1].end_time
            run_total_time = sum(seg.integrate_segment(seg.start_time, seg.end_time) for seg in run_segments)
            if isinstance(alignment_target, MetricPhaseTarget):
                candidate_end_beats = alignment_target.get_nearest_matching_beats(run_end_beat)
            else:
                candidate_end_beats = (alignment_target,)
            # snapshot the run's beats/curvature so we can restore between candidate attempts
            backup = [(seg.start_time, seg.end_time, seg.curve_shape) for seg in run_segments]
            for goal_end_beat in candidate_end_beats:
                # for each candidate end beat, we start by scaling the whole run by the appropriate factor
                # to hit that end beat. First determine that factor with a simple proportion.
                proportional_adjustment = (goal_end_beat - run_start_beat) / (run_end_beat - run_start_beat)
                # then loop through the segments with a running beat counter and rescale them
                b = run_start_beat
                for seg in run_segments:
                    old_dur = seg.duration
                    seg.start_time = b
                    seg.end_time = b = b + proportional_adjustment * old_dur
                # having gotten the beats to where we want them, we then attempt to use curvature to
                # adjust the total time back to its original value.
                if self._adjust_segments_time_duration(run_segments, run_total_time):
                    return True
                # on failure, restore the segments to their original state and try the next candidate
                for seg, (start, end, curve_shape) in zip(run_segments, backup):
                    seg.start_time, seg.end_time, seg.curve_shape = start, end, curve_shape
            # none of the candidates worked, return False
            return False

    ##################################################################################################################
    #                                           Functions and Envelope Loops
    ##################################################################################################################

    @tempo_modification
    def apply_function(self, func: Callable[[float], float], domain_start: float = 0, domain_end: float = None,
                       units: str = "tempo", duration_units: str = "beats", truncate: bool = False,
                       loop: bool = False, extension_increment: float = 2.0, scanning_step_size: float = 0.05,
                       key_point_resolution_multiple: int = 2, iterations: int = 6,
                       min_key_point_distance: float = 1e-7) -> None:
        """
        Apply a function for this TempoEnvelope to follow. If domain_end is None, this causes us to follow this
        function indefinitely; if a value is given for domain_end, then this appends an envelope that matches the
        contour of the prescribed function over a finite domain.

        :param func: a function (likely a lambda function) that maps a given beat/time to the desired
            tempo/rate/beatlength at that moment. See the `units` and `duration_units` arguments, which allow you
            to specify the units of the input and output of the function.
        :param domain_start: what part of the function's domain to start at (defaults to 0)
        :param domain_end: what part of the function's domain to end at (defaults to None, which causes the TempoHistory
            to follow the function indefinitely, or until :func:`stop_follow_function_or_envelope_loop` is called.)
        :param units: either "beats" or "time". If beats, then the `func` param specifies a mapping from beats to
            tempo/rate/beatlength; if time, then it specifies a mapping from sections to tempo/rate/beatlength.
        :param duration_units: one of ("tempo", "rate", "beatlength"). This determines the units of the output of the
            `func` parameter.
        :param truncate: if True, truncate any current projection of the TempoHistory into the future, and begin
            following this function from the current moment. If False, start following the function from the end of
            the current projection of the TempoHistory.
        :param loop: Only relevant if domain_end is set. If so, this determines whether or not we loop the segment
            of function.
        :param extension_increment: Only relevant if domain_end is None. When following the tempo/rate/beatlength
            function indefinitely, we project the function this far into the future, and only extend further as we
            reach the end of what we have projected.
        :param scanning_step_size: see :func:`expenvelope.Envelope.from_function`
        :param key_point_resolution_multiple: see :func:`expenvelope.Envelope.from_function`
        :param iterations: see :func:`expenvelope.Envelope.from_function`
        :param min_key_point_distance: see :func:`expenvelope.Envelope.from_function`
        """
        units = TempoUnits(units)
        duration_units = DurationUnits(duration_units)
        # truncate removes any segments that extend into the future
        if truncate:
            self.remove_segments_after(self.beat)
        # make sure that we're caught up to the current beat
        self.extend_to(self.beat)

        if domain_end is None:
            # set the function follow info
            self.follow_func_or_envelope_loop = FunctionFollowInfo(
                func, domain_start, extension_increment, units, duration_units,
                scanning_step_size, min_key_point_distance, iterations, key_point_resolution_multiple,
                self.end_time(), self.time + self.integrate_interval(self.beat, self.end_time())
            )
            # and then extend the follow function
            self._extend_follow_function()
        else:
            envelope = TempoEnvelope.from_function(
                func, domain_start, domain_end, units=units, duration_units=duration_units,
                scanning_step_size=scanning_step_size, min_key_point_distance=min_key_point_distance,
                iterations=iterations, key_point_resolution_multiple=key_point_resolution_multiple
            )

            self.append_envelope(envelope, loop=loop, truncate=truncate)

    def _extend_follow_function(self):
        """Extend the function that we're following by the prescribed step size."""
        func_envelope = TempoEnvelope.from_function(
            self.follow_func_or_envelope_loop.func, self.follow_func_or_envelope_loop.next_domain_start,
            self.follow_func_or_envelope_loop.next_domain_start + self.follow_func_or_envelope_loop.extension_increment,
            units=self.follow_func_or_envelope_loop.units, duration_units=self.follow_func_or_envelope_loop.duration_units,
            scanning_step_size=self.follow_func_or_envelope_loop.scanning_step_size,
            min_key_point_distance=self.follow_func_or_envelope_loop.min_key_point_distance,
            iterations=self.follow_func_or_envelope_loop.iterations,
            key_point_resolution_multiple=self.follow_func_or_envelope_loop.key_point_resolution_multiple
        )
        self.follow_func_or_envelope_loop.current_end_beat += func_envelope.end_time()
        self.follow_func_or_envelope_loop.current_end_time += func_envelope.integrate_interval(
            func_envelope.start_time(), func_envelope.end_time())
        self.follow_func_or_envelope_loop.next_domain_start += self.follow_func_or_envelope_loop.extension_increment
        self.append_envelope(func_envelope)

    def _extend_envelope_loop(self):
        """Extend the envelope loop by appending the looping envelope once."""
        self.follow_func_or_envelope_loop.current_end_beat += self.follow_func_or_envelope_loop.length_in_beats
        self.follow_func_or_envelope_loop.current_end_time += self.follow_func_or_envelope_loop.length_in_time
        self.append_envelope(self.follow_func_or_envelope_loop.env)

    def _extend_function_or_envelope_loop(self):
        """If we've applied a function or a looping envelope, extend it by one step size/copy"""
        if isinstance(self.follow_func_or_envelope_loop, FunctionFollowInfo):
            self._extend_follow_function()
        else:
            self._extend_envelope_loop()

    def stop_follow_function_or_envelope_loop(self):
        """If we've applied a function or a looping envelope, this causes us to stop doing so"""
        self.follow_func_or_envelope_loop = None
        self.truncate()

    ##################################################################################################################
    #                                                Advancing Time
    ##################################################################################################################

    def advance(self, duration: float, duration_units: DurationUnits = "beats") -> tuple[float, float]:
        """
        Advance the current beat/time in the envelope by the given number of beats.

        :param duration: how many beats or seconds to advance by
        :param duration_units: one of ("beats", "time")
        :return: tuple of delta beats, delta time
        """
        if duration_units == DurationUnits.BEATS:
            new_beat = snap_float_to_nice_decimal(self._beat + duration)
            new_time = self.time_at_beat(new_beat)
        else:
            new_time = snap_float_to_nice_decimal(self._t + duration)
            new_beat = self.beat_at_time(new_time)
        # it's important to first calculate both new beat and new time before setting the new values, because
        # function `self.beat_at_time` actually uses self.beat and self.time, which could otherwise be out of sync
        delta_beat, delta_time = new_beat - self._beat, new_time - self._t
        self._beat, self._t = new_beat, new_time
        return delta_beat, delta_time

    def go_to_beat(self, b: float) -> 'TempoEnvelope':
        """
        Jump straight to the given beat in this TempoHistory

        :param b: the beat to jump to
        :return: self, for chaining purposes
        """
        self._beat = snap_float_to_nice_decimal(b)
        self._t = snap_float_to_nice_decimal(self.integrate_interval(0, b))
        self.beat_at_time.cache_clear()
        self.time_at_beat.cache_clear()
        return self

    ##################################################################################################################
    #                                                   Utilities
    ##################################################################################################################

    @tempo_modification
    def truncate_at(self, beat: float) -> 'TempoEnvelope':
        return super().truncate_at(beat)

    def truncate(self) -> 'TempoEnvelope':
        """
        Removes all segments after the current beat.

        :return: self, for chaining purposes
        """
        return self.truncate_at(self._beat)

    def show_plot(self, title=None, resolution=25, show_segment_divisions=True, units="tempo",
                  x_range=None, y_range=None, show_current_beat=True):
        """
        Shows a plot of this TempoHistory using matplotlib.

        :param title: see :func:`TempoEnvelope.show_plot`
        :param resolution: see :func:`TempoEnvelope.show_plot`
        :param show_segment_divisions: see :func:`TempoEnvelope.show_plot`
        :param units: see :func:`TempoEnvelope.show_plot`
        :param x_range: see :func:`TempoEnvelope.show_plot`
        :param y_range: see :func:`TempoEnvelope.show_plot`
        :param show_current_beat: whether or not to show the current beat as a dashed line
        """
        title = "Graph of TempoHistory" if title is None else title
        plt = self._construct_plot(
            "Graph of TempoHistory" if title is None else title,
            resolution, show_segment_divisions, units,
            (min(0.0, self.start_time()), max(self.end_time(), self.beat)) if x_range is None else x_range, y_range
        )

        if show_current_beat:
            plt.vlines(self.beat, *plt.ylim(), colors="green", linestyles="dashed")
        plt.show()

    def as_tempo_envelope(self) -> TempoEnvelope:
        """
        Converts this TempoHistory to a simpler TempoEnvelope (removing reference to current beat and time)
        """
        return TempoEnvelope(self.levels, self.durations, self.curve_shapes, TempoUnits.BEATLENGTH)

    def __repr__(self):
        return "TempoHistory({}, {}, {}{})".format(
            TempoEnvelope.convert_units(self.levels, TempoUnits.BEATLENGTH, TempoUnits.TEMPO), self.durations,
            self.curve_shapes, ", beat={}".format(self._beat) if self._beat != 0 else ""
        )
