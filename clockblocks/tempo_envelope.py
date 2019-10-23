from expenvelope import Envelope, EnvelopeSegment
from copy import deepcopy
from .utilities import snap_float_to_nice_decimal, current_clock
import logging
import math


class TempoEnvelope(Envelope):

    def __init__(self, initial_rate_or_segments=1.0):
        # This is built on a envelope of beat length (units = s / beat, or really parent_beats / beat)
        if isinstance(initial_rate_or_segments, list):
            # this allows for compatibility with the Envelope constructor, helping class methods to work appropriately
            assert all(isinstance(x, EnvelopeSegment) for x in initial_rate_or_segments)
            super().__init__(initial_rate_or_segments)
        else:
            super().__init__()
            self.initialize(1 / initial_rate_or_segments)
        self._t = 0.0
        self._beat = 0.0

    def time(self):
        return self._t

    def beat(self):
        return self._beat

    @property
    def beat_length(self):
        return self.value_at(self._beat)

    @beat_length.setter
    def beat_length(self, beat_length):
        self.truncate()
        self.append_segment(beat_length, 0)

    def _bring_up_to_date(self, beat=None):
        beat = self.beat() if beat is None else beat
        # brings us up-to-date by adding a constant segment in case we haven't had a segment for a while
        if self.length() < beat:
            # no explicit segments have been made for a while, insert a constant segment to bring us up to date
            self.append_segment(self.end_level(), beat - self.length())
        return self

    def truncate(self, beat=None):
        # removes all segments after beat (which defaults to the current beat) and adds a constant segment
        # if necessary to being us up to that beat
        beat = self.beat() if beat is None else beat
        self.remove_segments_after(beat)
        self._bring_up_to_date(beat)
        return self

    def beat_length_at(self, beat, from_left=False):
        return self.value_at(beat, from_left)

    @property
    def rate(self):
        return 1 / self.beat_length

    @rate.setter
    def rate(self, rate):
        self.beat_length = 1/rate

    def rate_at(self, beat, from_left=False):
        return 1 / self.beat_length_at(beat, from_left)

    @property
    def tempo(self):
        return self.rate * 60

    @tempo.setter
    def tempo(self, tempo):
        self.rate = tempo / 60

    def tempo_at(self, beat, from_left=False):
        return self.rate_at(beat, from_left) * 60

    def set_beat_length_target(self, beat_length_target, duration, curve_shape=0, metric_phase_target=None,
                               duration_units="beats", truncate=True):
        if duration_units not in ("beats", "time"):
            raise ValueError("Argument duration_units must be either \"beat\" or \"time\".")
        if metric_phase_target is not None:
            metric_phase_target = MetricPhaseTarget.interpret(metric_phase_target)

        # truncate removes any segments that extend into the future
        if truncate:
            self.remove_segments_after(self.beat())
        # add a flat segment up to the current beat if needed
        self._bring_up_to_date()

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

    def set_rate_target(self, rate_target, duration, curve_shape=0, metric_phase_target=None,
                        duration_units="beats", truncate=True):
        self.set_beat_length_target(1 / rate_target, duration, curve_shape, metric_phase_target,
                                    duration_units, truncate)

    def set_tempo_target(self, tempo_target, duration, curve_shape=0, metric_phase_target=None,
                         duration_units="beats", truncate=True):
        self.set_beat_length_target(60 / tempo_target, duration, curve_shape, metric_phase_target,
                                    duration_units, truncate)

    def set_metric_phase_target_at_beat(self, beat, metric_phase_target):
        """
        Sets the goal phase in terms of time at the given beat. So, for instance, if we called
        set_metric_phase_target_at_beat(5, 0.5), this would mean that we want to be at time 1.5, 2.5, 3.5 etc. at
        beat 5. If we called set_metric_phase_target_at_beat(7, 1.25, 3), this would mean that at beat 7, we would want
        to be at time 1.25, 4.25, 7.25, etc.

        :param beat: The beat at which to have the given phase in time
        :param metric_phase_target: either a MetricPhaseTarget, or the argument to construct one
        :return True, if the adjustment is possible, False if not
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

    def adjust_time_at_beat(self, beat_to_adjust, desired_time):
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
        # ranges of how long each segment could take by adjusting curvature
        segment_time_ranges = [segment.get_integral_range() for segment in adjustable_segments]
        # range of how long the entire thing could take
        total_time_range = (sum(x[0] for x in segment_time_ranges), sum(x[1] for x in segment_time_ranges))
        # how long we want it to take
        goal_total_time = desired_time - self.time()

        # check if it's even possible to get to the desired time by simply adjusting curvatures
        if not total_time_range[0] < goal_total_time < total_time_range[1]:
            # if not return false, and go back to the backup segments (before we interpolated some points)
            self.segments = back_up
            return False

        # how long each segment currently takes
        segment_times = [segment.integrate_segment(segment.start_time, segment.end_time)
                         for segment in adjustable_segments]
        # how long all the segments take
        total_time = sum(segment_times)

        # on the off-chance that it already works perfectly, just return True
        if goal_total_time == total_time:
            self.segments = back_up  # also no need for those interpolations
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
        for segment, segment_time, segment_adjustment in zip(adjustable_segments, segment_times, segment_adjustments):
            segment.set_curvature_to_desired_integral(segment_time + segment_adjustment)
        return True

    def set_metric_phase_target_at_time(self, target_time, metric_phase_target):
        """
        Sets the goal phase in terms of beat at the given time. So, for instance, if we called
        set_metric_phase_target_at_beat(5, 0.5), this would mean that we want to be at beat 1.5, 2.5, 3.5 etc. at
        time 5. If we called set_metric_phase_target_at_beat(7, 1.25, 3), this would mean that at time 7, we would want
        to be at beat 1.25, 4.25, 7.25, etc.

        :param target_time: The time at which to have the given phase in beat
        :param metric_phase_target: either a MetricPhaseTarget, or the argument to construct one
        :return True, if the adjustment is possible, False if not
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

    def adjust_beat_at_time(self, time_to_adjust, desired_beat):
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

    def get_wait_time(self, beats):
        return self.integrate_interval(self._beat, self._beat + beats)

    def advance(self, beats, wait_time=None):
        if wait_time is None:
            wait_time = self.get_wait_time(beats)
        self._beat = snap_float_to_nice_decimal(self._beat + beats)
        self._t = snap_float_to_nice_decimal(self._t + wait_time)
        return beats, wait_time

    def get_beat_wait_from_time_wait(self, seconds):
        beat_to_get_to = self.get_upper_integration_bound(self._beat, seconds, max_error=0.00000001)
        return beat_to_get_to - self._beat

    def advance_time(self, seconds):
        beats = self.get_beat_wait_from_time_wait(seconds)
        self.advance(beats)
        return beats, seconds

    def go_to_beat(self, b):
        self._beat = snap_float_to_nice_decimal(b)
        self._t = snap_float_to_nice_decimal(self.integrate_interval(0, b))
        return self

    @staticmethod
    def convert_units(values, input_units, output_units):
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

        # this is a little confusing, like everything else about this function, but t represents the start beat of
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
    def from_levels_and_durations(cls, levels=(0, 0), durations=(0,), curve_shapes=None, offset=0,
                                  units="tempo", duration_units="beats"):
        assert duration_units in ("beats", "time"), "Duration units must be either \"beat\" or \"time\"."
        out_envelope = super().from_levels_and_durations(TempoEnvelope.convert_units(levels, units, "beatlength"),
                                                         durations, curve_shapes, offset)
        if duration_units == "time":
            return out_envelope.convert_durations_to_times()
        else:
            return out_envelope

    @classmethod
    def from_levels(cls, levels, length=1.0, offset=0, units="tempo", duration_units="beats"):
        assert duration_units in ("beats", "time"), "Duration units must be either \"beat\" or \"time\"."
        out_envelope = super().from_levels(TempoEnvelope.convert_units(levels, units, "beatlength"),
                                           length=length, offset=offset)
        if duration_units == "time":
            return out_envelope.convert_durations_to_times()
        else:
            return out_envelope

    @classmethod
    def from_list(cls, constructor_list, units="tempo", duration_units="beats"):
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
    def from_points(cls, *points, units="tempo", duration_units="beats"):
        assert all(len(point) >= 2 for point in points)
        assert duration_units in ("beats", "time"), "Duration units must be either \"beat\" or \"time\"."
        out_envelope = super().from_points([(t, TempoEnvelope.convert_units(l, units, "beatlength"))
                                            for (t, l) in points])
        if duration_units == "time":
            return out_envelope.convert_durations_to_times()
        else:
            return out_envelope

    @classmethod
    def from_function(cls, function, domain_start=0, domain_end=1, resolution_multiple=2,
                      key_point_precision=100, key_point_iterations=5, units="tempo", duration_units="beats"):
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
        # if we're past the end of the envelope, we want to plot that as a final constant segment
        # bring_up_to_date adds that segment, but we don't want to modify the original envelope, so we deepcopy
        env_to_plot = deepcopy(self)._bring_up_to_date() if self.end_time() < self.beat() else self

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Could not find matplotlib, which is needed for plotting.")

        fig, ax = plt.subplots()
        x_values, y_values = env_to_plot.get_graphable_point_pairs(resolution)
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

    def __init__(self, phase_or_phases, divisor=1, relative=False):
        self.phases = (phase_or_phases, ) if not hasattr(phase_or_phases, "__len__") else phase_or_phases
        if not all(0 <= x < divisor for x in self.phases):
            raise ValueError("One or more phases out of range for divisor.")
        self.divisor = divisor
        self.relative = relative

    @classmethod
    def interpret(cls, value):
        """
        Turn an input in the form of a tuple or just a number into a MetricPhaseTarget. E.g. we want the user to be able
        to hand in a tuple like (0.5, 3) and have it get interpreted as a target of 0.5 with divisor 3.

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

    def _get_nearest_matches(self, t, offset=0):
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
        return closest_below, closest_above if min_dist_below <= min_dist_above else closest_above, closest_below

    def get_nearest_matching_beats(self, beat):
        if self.relative:
            return self._get_nearest_matches(beat, current_clock().beat())
        else:
            return self._get_nearest_matches(beat)

    def get_nearest_matching_times(self, time):
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
