"""
Module defining the :class:`TempoEnvelope` class for describing a time-varying tempo, as well as the
:class:`MetricPhaseTarget` class, which specifies a goal arrival point within the beat (or meter) cycle.
"""

#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  #
#  SCAMP (Suite for Computer-Assisted Music in Python)                                           #
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

from expenvelope import Envelope, EnvelopeSegment
from copy import deepcopy
from .utilities import snap_float_to_nice_decimal, current_clock
import logging
import math
from typing import Union, Sequence, Tuple


class TempoEnvelope(Envelope):
    r"""
    A subclass of :class:`~expenvelope.envelope.Envelope` that is specifically designed for representing changing tempo
    curves. The underlying envelope represents beat length as a function of the current beat. A TempoEnvelopes also has
    a notion of the current beat and time and has methods for advancing time.

    :param initial_rate_or_segments: the rate to initialize this tempo envelope with, or a list of
        :class:`~expenvelope.envelope_segment.EnvelopeSegment`\ s to initialize it with. (Note that these need
        to represent beat length as a function of the current beat.)
    """

    def __init__(self, initial_rate_or_segments: Union[float, Sequence[EnvelopeSegment]] = 1.0):

        # This is built on a envelope of beat length (units = s / beat, or really parent_beats / beat)
        if isinstance(initial_rate_or_segments, list):
            # this allows for compatibility with the Envelope constructor, helping class methods to work appropriately
            assert all(isinstance(x, EnvelopeSegment) for x in initial_rate_or_segments)
            super().__init__(initial_rate_or_segments)
        else:
            super().__init__()
            self._initialize((1 / initial_rate_or_segments, ))
        self._t = 0.0
        self._beat = 0.0

    def time(self):
        """
        The current time. Time is found by integrating under the beatlength curve: seconds/beat * beats = seconds.
        """
        return self._t

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
        return self.value_at(self._beat)

    @beat_length.setter
    def beat_length(self, beat_length):
        self.truncate()
        self.append_segment(beat_length, 0)

    def _bring_up_to_date(self, beat: float = None) -> 'TempoEnvelope':
        beat = self.beat() if beat is None else beat
        # brings us up-to-date by adding a constant segment in case we haven't had a segment for a while
        if self.length() < beat:
            # no explicit segments have been made for a while, insert a constant segment to bring us up to date
            self.append_segment(self.end_level(), beat - self.length())
        return self

    def truncate(self, beat: float = None) -> 'TempoEnvelope':
        """
        Removes all segments after beat (which defaults to the current beat) and adds a constant segment if necessary
        to being us up to that beat.

        :param beat: the beat that we are truncating the tempo envelope after
        :return: self, for chaining purposes
        """
        beat = self.beat() if beat is None else beat
        self.remove_segments_after(beat)
        self._bring_up_to_date(beat)
        return self

    def beat_length_at(self, beat: float, from_left: bool = False) -> float:
        """
        Get the beat length at the given beat. If the beat length jumps at the given beat, the default is to return the
        beat length after the jump, though this can be overridden with the `from_left` argument.

        :param beat: the beat at which to get the beat length
        :param from_left: whether to evaluate from the right or left-hand side of the beat in question
        """
        return self.value_at(beat, from_left)

    @property
    def rate(self):
        """
        The current rate in beats/second
        """
        return 1 / self.beat_length

    @rate.setter
    def rate(self, rate):
        self.beat_length = 1/rate

    def rate_at(self, beat: float, from_left: bool = False) -> float:
        """
        Get the beat rate (in beats/second) at the given beat. If the rate jumps at the given beat, the default is to
        return the rate after the jump, though this can be overridden with the `from_left` argument.

        :param beat: the beat at which to get the rate
        :param from_left: whether to evaluate from the right or left-hand side of the beat in question
        """
        return 1 / self.beat_length_at(beat, from_left)

    @property
    def tempo(self):
        """
        The current tempo in beats/minute
        """
        return self.rate * 60

    @tempo.setter
    def tempo(self, tempo):
        self.rate = tempo / 60

    def tempo_at(self, beat: float, from_left: bool = False) -> float:
        """
        Get the tempo (in beats/minute) at the given beat. If the tempo jumps at the given beat, the default is to
        return the tempo after the jump, though this can be overridden with the `from_left` argument.

        :param beat: the beat at which to get the tempo
        :param from_left: whether to evaluate from the right or left-hand side of the beat in question
        """
        return self.rate_at(beat, from_left) * 60

    ##################################################################################################################
    #                                                Tempo Changes
    ##################################################################################################################

    def set_beat_length_target(self, beat_length_target: float, duration: float, curve_shape: float = 0,
                               metric_phase_target: Union[float, 'MetricPhaseTarget', Tuple] = None,
                               duration_units: str = "beats", truncate: bool = True) -> None:
        """
        Set a target beat length for this TempoEnvelope to reach in duration beats/seconds (with the unit defined by
        duration_units).

        :param beat_length_target: The beat length we want to reach
        :param duration: How long until we reach that beat length
        :param curve_shape: > 0 makes change happen later, < 0 makes change happen sooner
        :param metric_phase_target: This argument lets us align the arrival at the given beat length with a particular
            part of the parent beat (time), or, if we specified "time" as our duration units, it allows us to align
            the arrival at that specified time with a particular part of this clock's beat. This argument takes either
            a float in [0, 1), a MetricPhaseTarget object, or a tuple of arguments to the MetricPhaseTarget constructor
        :param duration_units: one of ("beats", "time"); defines whether the duration is in beats or in seconds.
        :param truncate: Whether or not to truncate this TempoEnvelope to the current beat before setting this target.
        """
        if duration_units not in ("beats", "time"):
            raise ValueError("Argument duration_units must be either \"beat\" or \"time\".")
        if metric_phase_target is not None:
            metric_phase_target = MetricPhaseTarget.interpret(metric_phase_target)

        # truncate removes any segments that extend into the future
        if truncate:
            self.remove_segments_after(self.beat())
        # add a flat segment up to the current beat if needed
        self._bring_up_to_date()

        self._add_segment(beat_length_target, duration, curve_shape, metric_phase_target, duration_units)

    def _add_segment(self, beat_length_target: float, duration: float, curve_shape: float = 0,
                     metric_phase_target: Union[float, 'MetricPhaseTarget', Tuple] = None,
                     duration_units: str = "beats") -> None:
        """
        The guts of adding a new segment, minus argument checking and truncating/bringing up to date.
        """
        if duration_units == "beats":
            extension_into_future = self.length() - self.beat()
            if duration < extension_into_future:
                raise ValueError("Duration to target must extend beyond the last existing target.")
            self.append_segment(beat_length_target, duration - extension_into_future, curve_shape)
            if metric_phase_target is not None:
                if not self._adjust_segment_end_time_to_metric_phase_target(self.segments[-1], metric_phase_target):
                    logging.warning("Metric phase target {} was not reachable".format(metric_phase_target))
        else:
            # units == "time", so we need to figure out how many beats are necessary
            time_extension_into_future = self.integrate_interval(self.beat(), self.length())
            if duration < time_extension_into_future:
                raise ValueError("Duration to target must extend beyond the last existing target.")

            # normalized_time = how long the curve would take if it were one beat long
            normalized_time = EnvelopeSegment(
                0, 1, self.value_at(self.length()), beat_length_target, curve_shape
            ).integrate_segment(0, 1)
            desired_curve_length = duration - time_extension_into_future
            self.append_segment(beat_length_target, desired_curve_length / normalized_time, curve_shape)
            if metric_phase_target is not None:
                if not self._adjust_segment_end_beat_to_metric_phase_target(self.segments[-1], metric_phase_target):
                    logging.warning("Metric phase target {} was not reachable".format(metric_phase_target))

    def set_beat_length_targets(self, beat_length_targets: Sequence[float], durations: Sequence[float],
                                curve_shapes: Sequence[float] = None,
                                metric_phase_targets: Sequence[Union[float, 'MetricPhaseTarget', Tuple]] = None,
                                duration_units: str = "beats", truncate: bool = True) -> None:
        """
        Same as set_beat_length_target, except that you can set multiple targets at once by providing lists to each
        of the arguments.

        :param beat_length_targets: list of the target beat_lengths
        :param durations: list of segment durations (in beats or seconds, as defined by duration_units)
        :param curve_shapes: list of segment curve_shapes (or none to not set curve shape)
        :param metric_phase_targets: list of metric phase targets for each segment (or None to ignore metric phase)
        :param duration_units: one of ("beats", "time"); defines whether the duration is in beats or in
            seconds/parent beats.
        :param truncate: Whether or not to truncate this TempoEnvelope to the current beat before setting these targets.
        """
        num_targets = len(beat_length_targets)
        if duration_units not in ("beats", "time"):
            raise ValueError("Argument duration_units must be either \"beat\" or \"time\".")
        curve_shapes = [0] * num_targets if curve_shapes is None else curve_shapes
        if len(durations) != num_targets:
            raise ValueError("Inconsistent number of targets and durations.")
        if len(curve_shapes) != num_targets:
            raise ValueError("Inconsistent number of targets and curve_shapes.")
        if metric_phase_targets is not None and len(metric_phase_targets) != num_targets:
            raise ValueError("Inconsistent number of metric phase targets and curve_shapes.")

        # truncate removes any segments that extend into the future
        if truncate:
            self.remove_segments_after(self.beat())
        # add a flat segment up to the current beat if needed
        self._bring_up_to_date()

        if metric_phase_targets is None:
            # no segments have phase targets, so it's simple
            for beat_length_target, duration, curve_shape in zip(beat_length_targets, durations, curve_shapes):
                self._add_segment(beat_length_target, duration, curve_shape, None, duration_units)
        else:
            metric_phase_targets = [(MetricPhaseTarget.interpret(x) if x is not None else None)
                                    for x in metric_phase_targets]
            # This is used to adjust metric phase, if desired. We keep track of all the segments
            # we've added since we last adjusted the metric phase.
            segments_to_adjust = []
            # We also keep track of the start and end beat/time of the current group of segments so that
            # we don't have to recalculate it all the time
            current_group_start_beat = current_group_end_beat = self.end_time()
            current_group_start_time = current_group_end_time = \
                self.time() + self.integrate_interval(self.beat(), self.end_time())

            for beat_length_target, duration, curve_shape, metric_phase_target in \
                    zip(beat_length_targets, durations, curve_shapes, metric_phase_targets):
                if metric_phase_target is None:
                    # no metric phase target for this segment, but some segments do have metric phase targets
                    # so we add it to our list of segments to adjust when we next have to adjust to a target
                    self._add_segment(beat_length_target, duration, curve_shape, metric_phase_target, duration_units)
                    added_segment = self.segments[-1]
                    segments_to_adjust.append(added_segment)
                    current_group_end_beat += added_segment.duration
                    current_group_end_time += added_segment.integrate_segment(added_segment.start_time,
                                                                              added_segment.end_time)
                else:
                    # if we're here then there is a metric phase target for the end of this segment
                    if len(segments_to_adjust) == 0:
                        # if we haven't built up any segments to adjust, then just add this one segment,
                        # adjusting it in the process
                        self._add_segment(beat_length_target, duration, curve_shape, metric_phase_target,
                                          duration_units)
                        added_segment = self.segments[-1]
                        current_group_end_beat += added_segment.duration
                        current_group_end_time += added_segment.integrate_segment(added_segment.start_time,
                                                                                  added_segment.end_time)
                        current_group_start_beat = current_group_end_beat
                        current_group_start_time = current_group_end_time
                        continue
                    # Otherwise, we add the segment without adjusting it in the process...
                    self._add_segment(beat_length_target, duration, curve_shape, None, duration_units)
                    added_segment = self.segments[-1]
                    segments_to_adjust.append(added_segment)
                    current_group_end_beat += added_segment.duration
                    current_group_end_time += added_segment.integrate_segment(added_segment.start_time,
                                                                              added_segment.end_time)

                    # ...and then we try to reach the target by adjusting all of the segments since the last adjustment
                    success = False  # did we successfully adjust?
                    if duration_units == "beats":
                        for goal_end_time in metric_phase_target.get_nearest_matching_times(current_group_end_time):
                            # try both the nearest matching time before and after
                            goal_time_duration = goal_end_time - current_group_start_time
                            if self._adjust_segments_time_duration(segments_to_adjust, goal_time_duration):
                                # if one of them works, declare success and break
                                current_group_end_time = goal_end_time  # reset the end time based on the adjustment
                                success = True
                                break
                    elif duration_units == "time":
                        for goal_end_beat in metric_phase_target.get_nearest_matching_beats(current_group_end_beat):
                            # try both the nearest matching beat before and after
                            # first, squeeze/stretch all the segments to take up an appropriate number of beats
                            proportional_adjustment = (goal_end_beat - current_group_start_beat) / \
                                                      (current_group_end_beat - current_group_start_beat)
                            b = current_group_start_beat
                            for segment in segments_to_adjust:
                                old_dur = segment.duration
                                segment.start_time = b
                                segment.end_time = b = b + proportional_adjustment * old_dur
                            # then try to re-adjust to get back to the original end time
                            if self._adjust_segments_time_duration(segments_to_adjust,
                                                                   current_group_end_time - current_group_start_time):
                                # if it works, declare success and break
                                current_group_end_beat = goal_end_beat  # reset the end beat based on the adjustment
                                success = True
                                break
                    if not success:
                        logging.warning("Metric phase target {} was not reachable.".format(metric_phase_target))
                    else:
                        # If it did succeed, clear the segments_to_adjust. We don't want to be adjusting any of the
                        # segments that we just adjusted, since they would get messed up.
                        segments_to_adjust.clear()
                        # also reset the group start and end beat/time
                        current_group_start_beat = current_group_end_beat
                        current_group_start_time = current_group_end_time

    def set_rate_target(self, rate_target: float, duration: float, curve_shape: float = 0,
                        metric_phase_target: Union[float, 'MetricPhaseTarget', Tuple] = None,
                        duration_units: str = "beats", truncate: bool = True) -> None:
        """
        Set a target beat rate for this TempoEnvelope to reach in duration beats/seconds (with the unit defined by
        duration_units).

        :param rate_target: The beat rate we want to reach
        :param duration: How long until we reach that beat rate
        :param curve_shape: > 0 makes change happen later, < 0 makes change happen sooner
        :param metric_phase_target: This argument lets us align the arrival at the given beat length with a particular
            part of the parent beat (time), or, if we specified "time" as our duration units, it allows us to align
            the arrival at that specified time with a particular part of this clock's beat. This argument takes either
            a float in [0, 1), a MetricPhaseTarget object, or a tuple of arguments to the MetricPhaseTarget constructor
        :param duration_units: one of ("beats", "time"); defines whether the duration is in beats or in seconds.
        :param truncate: Whether or not to truncate this TempoEnvelope to the current beat before setting this target.
        """
        self.set_beat_length_target(1 / rate_target, duration, curve_shape, metric_phase_target,
                                    duration_units, truncate)

    def set_rate_targets(self, rate_targets: Sequence[float], durations: Sequence[float],
                         curve_shapes: Sequence[float] = None,
                         metric_phase_targets: Sequence[Union[float, 'MetricPhaseTarget', Tuple]] = None,
                         duration_units: str = "beats", truncate: bool = True) -> None:
        """
        Same as set_rate_target, except that you can set multiple targets at once by providing lists to each
        of the arguments.

        :param rate_targets: list of the target beat rates
        :param durations: list of segment durations (in beats or seconds, as defined by duration_units)
        :param curve_shapes: list of segment curve_shapes (or none to not set curve shape)
        :param metric_phase_targets: list of metric phase targets for each segment (or None to ignore metric phase)
        :param duration_units: one of ("beats", "time"); defines whether the duration is in beats or in
            seconds/parent beats.
        :param truncate: Whether or not to truncate this TempoEnvelope to the current beat before setting these targets.
        """
        self.set_beat_length_targets([1 / x for x in rate_targets], durations, curve_shapes, metric_phase_targets,
                                     duration_units, truncate)

    def set_tempo_target(self, tempo_target: float, duration: float, curve_shape: float = 0,
                         metric_phase_target: Union[float, 'MetricPhaseTarget', Tuple] = None,
                         duration_units: str = "beats", truncate: bool = True) -> None:
        """
        Set a target tempo for this TempoEnvelope to reach in duration beats/seconds (with the unit defined by
        duration_units).

        :param tempo_target: The tempo we want to reach
        :param duration: How long until we reach that tempo
        :param curve_shape: > 0 makes change happen later, < 0 makes change happen sooner
        :param metric_phase_target: This argument lets us align the arrival at the given beat length with a particular
            part of the parent beat (time), or, if we specified "time" as our duration units, it allows us to align
            the arrival at that specified time with a particular part of this clock's beat. This argument takes either
            a float in [0, 1), a MetricPhaseTarget object, or a tuple of arguments to the MetricPhaseTarget constructor
        :param duration_units: one of ("beats", "time"); defines whether the duration is in beats or in seconds.
        :param truncate: Whether or not to truncate this TempoEnvelope to the current beat before setting this target.
        """
        self.set_beat_length_target(60 / tempo_target, duration, curve_shape, metric_phase_target,
                                    duration_units, truncate)

    def set_tempo_targets(self, tempo_targets: Sequence[float], durations: Sequence[float],
                          curve_shapes: Sequence[float] = None,
                          metric_phase_targets: Sequence[Union[float, 'MetricPhaseTarget', Tuple]] = None,
                          duration_units: str = "beats", truncate: bool = True) -> None:
        """
        Same as set_tempo_target, except that you can set multiple targets at once by providing lists to each
        of the arguments.

        :param tempo_targets: list of the target tempos
        :param durations: list of segment durations (in beats or seconds, as defined by duration_units)
        :param curve_shapes: list of segment curve_shapes (or none to not set curve shape)
        :param metric_phase_targets: list of metric phase targets for each segment (or None to ignore metric phase)
        :param duration_units: one of ("beats", "time"); defines whether the duration is in beats or in
            seconds/parent beats.
        :param truncate: Whether or not to truncate this TempoEnvelope to the current beat before setting these targets.
        """
        self.set_beat_length_targets([60 / x for x in tempo_targets], durations, curve_shapes, metric_phase_targets,
                                     duration_units, truncate)

    # ----------------------------------------- Metric Phase adjustments ---------------------------------------------

    # These two methods are for just adjusting a single segment's metric phase in beat or time.
    # They are used when adding single segments that we want to adjust the phase of

    def _adjust_segment_end_time_to_metric_phase_target(self, segment, metric_phase_target):
        # this is confusing; segment.start_time and segment.end_time are really the start and end *beats*
        segment_start_time = self.time() + self.integrate_interval(self.beat(), segment.start_time)
        segment_end_time = segment_start_time + segment.integrate_segment(segment.start_time, segment.end_time)
        for new_end_time in metric_phase_target.get_nearest_matching_times(segment_end_time):
            try:
                segment.set_curvature_to_desired_integral(new_end_time - segment_start_time)
                return True
            except ValueError:
                pass
        return False

    @staticmethod
    def _adjust_segment_end_beat_to_metric_phase_target(segment, metric_phase_target):
        # this is how long the segment currently takes; we want to end up with this being the same
        original_integral = segment.integrate_segment(segment.start_time, segment.end_time)
        for new_end_beat in metric_phase_target.get_nearest_matching_beats(segment.end_time):
            try:
                # shrink the segment
                segment.end_time = new_end_beat
                # and then try to change the curvature to return to the original time duration
                segment.set_curvature_to_desired_integral(original_integral)
                return True
            except ValueError:
                pass
        return False

    # These methods are used when we want to adjust the metric phase at the end of a group of segments.

    def adjust_metric_phase_at_beat(self, beat: float,
                                    metric_phase_target: Union[float, 'MetricPhaseTarget', Tuple]) -> bool:
        """
        Sets the goal (time) metric phase at the given beat. So, for instance, if we called
        ``adjust_metric_phase_at_beat(5, 0.5)``, this would mean that we want to be at time 1.5, 2.5, 3.5 etc. at
        beat 5. If we called ``adjust_metric_phase_at_beat(7, 1.25, 3)``, this would mean that at beat 7, we would
        want to be at time 1.25, 4.25, 7.25, etc.

        :param beat: The beat at which to have the given phase in time
        :param metric_phase_target: either a :class:`MetricPhaseTarget`, or the argument to construct one
        :return: True, if the adjustment is possible, False if not
        """
        if beat > self.length() or beat <= self.beat():
            raise ValueError("Cannot adjust metric phase before current beat or beyond the end of the TempoEnvelope")

        metric_phase_target = MetricPhaseTarget.interpret(metric_phase_target)

        # what's the current time at the beat?
        time_at_beat = self.time() + self.integrate_interval(self.beat(), beat)

        # try to adjust that to one of the nearby target phases
        for good_phase_time in metric_phase_target.get_nearest_matching_times(time_at_beat):
            if self.adjust_time_at_beat(beat, good_phase_time):
                # the adjustment worked (returned true), so return True to say that we succeeded
                return True

        # if we get here, neither adjustment was possible, so we failed. Return false.
        return False

    def adjust_time_at_beat(self, beat_to_adjust: float, desired_time: float) -> bool:
        """
        Adjusts the curvature of segments from now until beat so that we reach it at desired_time, if possible. If not
        possible, leaves the TempoCurve unchanged and returns False

        :param beat_to_adjust: the beat at which we want to be at a particular time
        :param desired_time: the time we want to be at
        :return: True if the adjustment worked, False if it's impossible
        """
        assert self.beat() < beat_to_adjust <= self.length()

        # make a copy of the original segments lists to fall back on in case we fail
        back_up = deepcopy(self.segments)
        self.insert_interpolated(self.beat())
        self.insert_interpolated(beat_to_adjust)
        adjustable_segments = self.segments[self._get_index_of_segment_at(self.beat(), right_most=True):
                                            self._get_index_of_segment_at(beat_to_adjust, left_most=True) + 1]
        goal_total_time = desired_time - self.time()
        result = TempoEnvelope._adjust_segments_time_duration(adjustable_segments, goal_total_time)

        if result == "no change":
            # it worked, but we didn't have to change anything
            # no there's no need for the interpolations
            self.segments = back_up
            return True
        elif result:
            # it worked, return True
            return True
        else:
            # the adjustment failed, so return to the old segments before interpolation
            # and return false to signal the failure
            self.segments = back_up
            return False

    @staticmethod
    def _adjust_segments_time_duration(which_segments: Sequence[EnvelopeSegment], goal_total_time: float):
        """
        Adjusts the total time that the segments take without changing the total beats

        :param which_segments: which segments to adjust.
        :param goal_total_time: the total time we want them to take
        :return: True if it's possible, False if not, and "no change" in the off-chance that no change was needed
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

        # on the off-chance that it already works perfectly, return "no change" to indicate that it worked,
        # but that it was totally unnecessary
        if goal_total_time == total_time:
            return "no change"

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

    def adjust_metric_phase_at_time(self, target_time: float,
                                    metric_phase_target: Union[float, 'MetricPhaseTarget', Tuple]) -> bool:
        """
        Sets the goal (beat) metric phase at the given time. So, for instance, if we called
        ``adjust_metric_phase_at_time(5, 0.5)``, this would mean that at time 5 we want to be at beat 1.5, 2.5, 3.5
        etc. If we called ``adjust_metric_phase_at_time(7, 1.25, 3)``, this would mean that at time 7, we would want
        to be at beat 1.25, 4.25, 7.25, etc.

        :param target_time: The time at which to have the given phase in beat
        :param metric_phase_target: either a MetricPhaseTarget, or the argument to construct one
        :return: True, if the adjustment is possible, False if not
        """

        envelope_end_time = self.time() + self.integrate_interval(self.beat(), self.end_time())
        if target_time > envelope_end_time or target_time <= self.time():
            raise ValueError("Cannot adjust metric phase before current beat or beyond the end of the TempoEnvelope")

        metric_phase_target = MetricPhaseTarget.interpret(metric_phase_target)

        # what's the current beat at the time?
        beat_at_time = self.beat() + self.get_beat_wait_from_time_wait(target_time - self.time())

        # try to adjust that to one of the nearby target phases
        for good_phase_beat in metric_phase_target.get_nearest_matching_beats(beat_at_time):
            if self.adjust_beat_at_time(target_time, good_phase_beat):
                # the adjustment worked (returned true), so return True to say that we succeeded
                return True

        # if we get here, neither adjustment was possible, so we failed. Return false.
        return False

    def adjust_beat_at_time(self, time_to_adjust: float, desired_beat: float) -> bool:
        """
        Adjusts the curvature of segments from now until the specified time so that we reach it at desired_beat,
        if possible. If not possible, leaves the TempoCurve unchanged and returns False.

        :param time_to_adjust: the time at which we want to be at a particular beat
        :param desired_beat: the beat we want to be at
        :return: True if the adjustment worked, False if it's impossible
        """
        envelope_end_time = self.time() + self.integrate_interval(self.beat(), self.end_time())
        assert self.time() < time_to_adjust <= envelope_end_time

        # make a copy of the original segments lists to fall back on in case we fail
        back_up = deepcopy(self.segments)
        current_beat_at_adjust_point = self.beat() + self.get_beat_wait_from_time_wait(time_to_adjust - self.time())

        start_beat = self.insert_interpolated(self.beat())
        # if the insertion does nothing because it's too close to an existing point, it will return the existing point
        current_beat_at_adjust_point = self.insert_interpolated(current_beat_at_adjust_point)

        adjustable_index_start = self._get_index_of_segment_at(self.beat(), right_most=True)
        adjustable_index_end = self._get_index_of_segment_at(current_beat_at_adjust_point, left_most=True) + 1
        adjustable_segments = self.segments[adjustable_index_start: adjustable_index_end]

        # first we squeeze or stretch all the segments so that we reach the right beat at the end of the last one
        delta_beat = desired_beat - current_beat_at_adjust_point
        proportional_length_adjustment = (desired_beat - start_beat) / (current_beat_at_adjust_point - start_beat)

        b = adjustable_segments[0].start_time
        for segment in adjustable_segments:
            old_dur = segment.duration
            segment.start_time = b
            segment.end_time = b = b + proportional_length_adjustment * old_dur

        for segment in self.segments[adjustable_index_end:]:
            segment.start_time += delta_beat
            segment.end_time += delta_beat

        # now that we squeezed or stretched so as to be at the correct moment in the curve, on the correct beat
        # see if we can adjust the curvature of the segments so that the time at that moment is unchanged
        if self.adjust_time_at_beat(desired_beat, time_to_adjust):
            # if it works, return True
            return True
        else:
            # otherwise, revert and return false
            self.segments = back_up
            return False

    ##################################################################################################################
    #                                                Advancing Time
    ##################################################################################################################

    def get_wait_time(self, beats: float) -> float:
        """
        Get the amount of time it would take to wait the specified number of beats starting at the current point
        in the TempoEnvelope.

        :param beats: how many beats to wait
        :return: how much time that will take
        """
        return self.integrate_interval(self._beat, self._beat + beats)

    def advance(self, beats: float) -> Tuple[float, float]:
        """
        Advance the current beat/time in the envelope by the given number of beats.

        :param beats: how many beats to advance by
        :return: tuple of delta beats, delta time
        """
        wait_time = self.get_wait_time(beats)
        self._beat = snap_float_to_nice_decimal(self._beat + beats)
        self._t = snap_float_to_nice_decimal(self._t + wait_time)
        return beats, wait_time

    def get_beat_wait_from_time_wait(self, seconds: float) -> float:
        """
        Get the amount of beats we would have to wait to wait for the given number of seconds, starting at the current
        point in the TempoEnvelope.

        :param seconds: how many seconds to wait
        :return: how many beats that would correspond to
        """
        beat_to_get_to = self.get_upper_integration_bound(self._beat, seconds, max_error=0.00000001)
        return beat_to_get_to - self._beat

    def advance_time(self, seconds: float):
        """
        Advance the current beat/time in the envelope by the given number of seconds.

        :param seconds: how many seconds to advance by
        :return: tuple of delta beats, delta time
        """
        beats = self.get_beat_wait_from_time_wait(seconds)
        self.advance(beats)
        return beats, seconds

    def go_to_beat(self, b: float) -> 'TempoEnvelope':
        """
        Jump straight to the given beat in this TempoEnvelope

        :param b: the beat to jump to
        :return: self, for chaining purposes
        """
        self._beat = snap_float_to_nice_decimal(b)
        self._t = snap_float_to_nice_decimal(self.integrate_interval(0, b))
        return self

    ##################################################################################################################
    #                                             Conversion Utilities
    ##################################################################################################################

    @staticmethod
    def convert_units(values: Union[float, Sequence[float]], input_units: str,
                      output_units: str) -> Union[float, Sequence[float]]:
        """
        Utility method to convert values between unites of tempo, rate and beat length.

        :param values: a list of values in terms of the input_units
        :param input_units: current units of the given values (either "tempo", "rate", or "beat length")
        :param output_units: desired units to convert to (either "tempo", "rate", or "beat length")
        :return: the list of values, converted to output units
        """
        in_units = input_units.lower().replace(" ", "")
        out_units = output_units.lower().replace(" ", "")
        assert in_units in ("tempo", "rate", "beatlength") and out_units in ("tempo", "rate", "beatlength"), \
            "Invalid value of {} for units. Must be \"tempo\", \"rate\" or \"beat length\"".format(input_units)
        if in_units == out_units:
            return values
        else:
            convert_input_to_beat_length = (lambda x: 1 / x) if in_units == "rate" \
                else (lambda x: 60 / x) if in_units == "tempo" else (lambda x: x)
            convert_beat_length_to_output = (lambda x: 1 / x) if out_units == "rate" \
                else (lambda x: 60 / x) if out_units == "tempo" else (lambda x: x)
            if hasattr(values, "__len__"):
                return tuple(convert_beat_length_to_output(convert_input_to_beat_length(x)) for x in values)
            else:
                return convert_beat_length_to_output(convert_input_to_beat_length(values))

    def convert_durations_to_times(self):
        """
        Warps this tempo_curve so that all of the locations of key points get re-interpreted as times instead of beat
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
    #                                                Class Methods
    ##################################################################################################################

    @classmethod
    def from_levels_and_durations(cls, levels: Sequence = (0, 0), durations: Sequence[float] = (0,),
                                  curve_shapes: Sequence[Union[float, str]] = None, offset: float = 0,
                                  units: str = "tempo", duration_units: str = "beats") -> 'TempoEnvelope':
        """
        Constructs a TempoEnvelope from the given levels, durations and curve shapes, using the specified units.

        :param levels: levels of the curve segments (i.e. tempo values) in the units specified by the `units` argument
        :param durations: durations of the curve segments in the units specified by the `duration_units` argument
        :param curve_shapes: see :func:`~expenvelope.envelope.Envelope.from_levels_and_durations`
        :param offset: beat offset for the start of this TempoEnvelope (inherited from Envelope, probably not useful)
        :param units: one of "tempo", "rate" or "beat length", determining how we interpret the levels given
        :param duration_units: either "beats" or "time", determining how we interpret the durations given
        :return: a TempoEnvelope, constructed accordingly
        """
        assert duration_units in ("beats", "time"), "Duration units must be either \"beat\" or \"time\"."
        out_envelope = super().from_levels_and_durations(TempoEnvelope.convert_units(levels, units, "beatlength"),
                                                         durations, curve_shapes, offset)
        if duration_units == "time":
            return out_envelope.convert_durations_to_times()
        else:
            return out_envelope

    @classmethod
    def from_levels(cls, levels: Sequence[float], length: float = 1.0, offset: float = 0, units: str = "tempo",
                    duration_units: str = "beats") -> 'TempoEnvelope':
        """
        Constructs a TempoEnvelope from the given levels and total length, using the specified units.

        :param levels: levels of the curve segments (i.e. tempo values) in the units specified by the `units` argument
        :param length: total length of the tempo curve, in the units specified by the `duration_units` argument
        :param offset: beat offset for the start of this TempoEnvelope (inherited from Envelope, probably not useful)
        :param units: one of "tempo", "rate" or "beat length", determining how we interpret the levels given
        :param duration_units: either "beats" or "time", determining how we interpret the durations given
        :return: a TempoEnvelope, constructed accordingly
        """
        assert duration_units in ("beats", "time"), "Duration units must be either \"beat\" or \"time\"."
        out_envelope = super().from_levels(TempoEnvelope.convert_units(levels, units, "beatlength"),
                                           length=length, offset=offset)
        if duration_units == "time":
            return out_envelope.convert_durations_to_times()
        else:
            return out_envelope

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
        assert all(len(point) >= 2 for point in points)
        assert duration_units in ("beats", "time"), "Duration units must be either \"beat\" or \"time\"."
        out_envelope = super().from_points([(t, TempoEnvelope.convert_units(l, units, "beatlength"))
                                            for (t, l) in points])
        if duration_units == "time":
            return out_envelope.convert_durations_to_times()
        else:
            return out_envelope

    @classmethod
    def from_function(cls, function, domain_start=0, domain_end=1, resolution_multiple=2, key_point_precision=100,
                      key_point_iterations=5, units: str = "tempo", duration_units: str = "beats") -> 'TempoEnvelope':
        """
        Constructs a TempoEnvelope that approximates an arbitrary function. The domain of the function is in units
        defined by the `duration_units` parameter, and the range is in units defined by the `units` parameter.

        :param function: A function from beat/time to tempo/rate/beat length, as defined by the `duration_units` and
            `units` parameters.
        :param domain_start: see :func:`~expenvelope.envelope.Envelope.from_function`
        :param domain_end: see :func:`~expenvelope.envelope.Envelope.from_function`
        :param resolution_multiple: see :func:`~expenvelope.envelope.Envelope.from_function`
        :param key_point_precision: see :func:`~expenvelope.envelope.Envelope.from_function`
        :param key_point_iterations: see :func:`~expenvelope.envelope.Envelope.from_function`
        :param units: one of "tempo", "rate" or "beat length", determining how we interpret the function output
        :param duration_units: either "beats" or "time", determining how we interpret the function input
        :return: a TempoEnvelope, constructed accordingly
        """
        assert duration_units in ("beats", "time"), "Duration units must be either \"beat\" or \"time\"."
        converted_function = (lambda x: TempoEnvelope.convert_units(function(x), units, "beatlength")) \
            if units.lower().replace(" ", "") != "beatlength" else function
        out_envelope = super().from_function(converted_function, domain_start, domain_end, resolution_multiple,
                                             key_point_precision, key_point_iterations)
        if duration_units == "time":
            return out_envelope.convert_durations_to_times()
        else:
            return out_envelope

    ##################################################################################################################
    #                                                Other Utilities
    ##################################################################################################################

    def show_plot(self, title=None, resolution=25, show_segment_divisions=True, units="tempo",
                  x_range=None, y_range=None):
        """
        Shows a plot of this TempoEnvelope using matplotlib.

        :param title: A title to give the plot.
        :param resolution: number of points to use per envelope segment
        :param show_segment_divisions: Whether or not to place dots at the division points between envelope segments
        :param units: one of "tempo", "rate" or "beat length", determining the units of the y-axis
        :param x_range: min and max value shown on the x-axis
        :param y_range: min and max value shown on the y-axis
        """
        # if we're past the end of the envelope, we want to plot that as a final constant segment
        # bring_up_to_date adds that segment, but we don't want to modify the original envelope, so we deepcopy
        env_to_plot = deepcopy(self)._bring_up_to_date() if self.end_time() < self.beat() else self

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Could not find matplotlib, which is needed for plotting.")

        fig, ax = plt.subplots()
        x_values, y_values = env_to_plot._get_graphable_point_pairs(resolution)
        ax.plot(x_values, TempoEnvelope.convert_units(y_values, "beat length", units))
        if show_segment_divisions:
            ax.plot(env_to_plot.times, TempoEnvelope.convert_units(env_to_plot.levels, "beat length", units), 'o')
        plt.xlabel("Beat")
        plt.ylabel("Tempo" if units == "tempo" else "Rate" if units == "rate" else "Beat Length")
        if x_range is not None:
            plt.xlim(x_range)
        if y_range is not None:
            plt.ylim(y_range)
        ax.set_title('Graph of TempoEnvelope' if title is None else title)
        plt.show()

    def __repr__(self):
        return "TempoEnvelope({}, {}, {})".format(self.levels, self.durations, self.curve_shapes)


class MetricPhaseTarget:

    """
    Class representing a particular point in a (beat or measure) cycle.

    :param phase_or_phases: Where we are in the cycle.
    :param divisor: Length of the cycle (defaults to one beat, meaning that this specifies where we are in the beat)
    :param relative: Whether or not the start of the cycle is measured relative to the current beat/time or to the
        start of the clock.
    """

    def __init__(self, phase_or_phases: Union[float, Sequence[float]], divisor: float = 1, relative: bool = False):
        self.phases = (phase_or_phases, ) if not hasattr(phase_or_phases, "__len__") else phase_or_phases
        if not all(0 <= x < divisor for x in self.phases):
            raise ValueError("One or more phases out of range for divisor.")
        self.divisor = divisor
        self.relative = relative

    @classmethod
    def interpret(cls, value: Union[float, Sequence]) -> 'MetricPhaseTarget':
        """
        Interpret a tuple or just a number as a MetricPhaseTarget. E.g. we want the user to be able to hand in a tuple
        like (0.5, 3) and have it get interpreted as a target of 0.5 with divisor 3.

        :param value: either a tuple (which becomes the constructor arguments), a MetricPhaseTarget (which
            is passed through unchanged), or a number which is treated as the phase with other args as defaults.
        :return: A MetricPhaseTarget, interpreted from the argument
        """
        if isinstance(value, MetricPhaseTarget):
            return value
        elif hasattr(value, "__len__"):
            return MetricPhaseTarget(*value)
        else:
            return MetricPhaseTarget(value)

    def _get_nearest_matches(self, t: float, offset: float = 0) -> Tuple[float, float]:
        floored_value = math.floor(t / self.divisor) * self.divisor
        closest_below = None
        closest_above = None
        min_dist_below = float("inf")
        min_dist_above = float("inf")
        for base_multiple in (floored_value - self.divisor, floored_value, floored_value + self.divisor):
            for remainder in self.phases:
                remainder = (remainder + offset) % self.divisor
                this_value = base_multiple + remainder
                if this_value < t and t - this_value < min_dist_below:
                    closest_below = this_value
                    min_dist_below = t - this_value
                elif this_value >= t and this_value - t < min_dist_above:
                    closest_above = this_value
                    min_dist_above = this_value - t
        # return the closest above and below in order of closeness
        return (closest_below, closest_above) if min_dist_below <= min_dist_above else (closest_above, closest_below)

    def get_nearest_matching_beats(self, beat: float) -> Tuple[float, float]:
        """
        Get the nearest beats below and above the given beat with the correct metric phase

        :param beat: the beat to search around
        :return: tuple of nearest beat below, nearest beat above
        """
        if self.relative:
            return self._get_nearest_matches(beat, current_clock().beat())
        else:
            return self._get_nearest_matches(beat)

    def get_nearest_matching_times(self, time: float) -> Tuple[float, float]:
        """
        Get the nearest times below and above the given time with the correct metric phase

        :param time: the time to search around
        :return: tuple of nearest time below, nearest time above
        """
        if self.relative:
            return self._get_nearest_matches(time, current_clock().time())
        else:
            return self._get_nearest_matches(time)

    def __repr__(self):
        return "MetricPhaseTarget({}{}{})".format(
            str(self.phases[0]) if hasattr(self.phases, "__len__") else self.phases,
            (", " + str(self.divisor)) if self.divisor != 1 else "",
            ", True" if self.relative else "",
        )
