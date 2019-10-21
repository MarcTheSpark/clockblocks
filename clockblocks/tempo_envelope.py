from expenvelope import Envelope, EnvelopeSegment
from copy import deepcopy
from .utilities import snap_float_to_nice_decimal


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
        self._beats = 0.0

    def time(self):
        return self._t

    def beats(self):
        return self._beats

    @property
    def beat_length(self):
        return self.value_at(self._beats)

    def bring_up_to_date(self, beat=None):
        beat = self.beats() if beat is None else beat
        # brings us up-to-date by adding a constant segment in case we haven't had a segment for a while
        if self.length() < beat:
            # no explicit segments have been made for a while, insert a constant segment to bring us up to date
            self.append_segment(self.end_level(), beat - self.length())
        return self

    def truncate(self, beat=None):
        # removes all segments after beat (which defaults to the current beat) and adds a constant segment
        # if necessary to being us up to that beat
        beat = self.beats() if beat is None else beat
        self.remove_segments_after(beat)
        self.bring_up_to_date(beat)
        return self

    @beat_length.setter
    def beat_length(self, beat_length):
        self.truncate()
        self.append_segment(beat_length, 0)

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

    def set_beat_length_target(self, beat_length_target, duration, curve_shape=0,
                               duration_units="beats", truncate=True):
        assert duration_units in ("beats", "time"), "Duration units must be either \"beat\" or \"time\"."
        # truncate removes any segments that extend into the future
        if truncate:
            self.remove_segments_after(self.beats())

        # brings us up-to-date by adding a constant segment in case we haven't had a segment for a while
        if self.length() < self.beats():
            # no explicit segments have been made for a while, insert a constant segment to bring us up to date
            self.append_segment(self.end_level(), self.beats() - self.length())

        if duration_units == "beats":
            extension_into_future = self.length() - self.beats()
            if duration < extension_into_future:
                raise ValueError("Duration to target must extend beyond the last existing target.")
            self.append_segment(beat_length_target, duration - extension_into_future, curve_shape)
        else:
            # units == "time", so we need to figure out how many beats are necessary
            time_extension_into_future = self.integrate_interval(self.beats(), self.length())
            if duration < time_extension_into_future:
                raise ValueError("Duration to target must extend beyond the last existing target.")

            # normalized_time = how long the curve would take if it were one beat long
            normalized_time = EnvelopeSegment(
                0, 1, self.value_at(self.length()), beat_length_target, curve_shape
            ).integrate_segment(0, 1)
            desired_curve_length = duration - time_extension_into_future
            self.append_segment(beat_length_target, desired_curve_length / normalized_time, curve_shape)

    def set_rate_target(self, rate_target, duration, curve_shape=0, duration_units="beats", truncate=True):
        self.set_beat_length_target(1 / rate_target, duration, curve_shape, duration_units, truncate)

    def set_tempo_target(self, tempo_target, duration, curve_shape=0, duration_units="beats", truncate=True):
        self.set_beat_length_target(60 / tempo_target, duration, curve_shape, duration_units, truncate)

    def get_wait_time(self, beats):
        return self.integrate_interval(self._beats, self._beats + beats)

    def advance(self, beats, wait_time=None):
        if wait_time is None:
            wait_time = self.get_wait_time(beats)
        self._beats += beats
        self._t = snap_float_to_nice_decimal(self._t + wait_time)
        return beats, wait_time

    def get_beat_wait_from_time_wait(self, seconds):
        beat_to_get_to = self.get_upper_integration_bound(self._beats, seconds, max_error=0.00000001)
        return beat_to_get_to - self._beats

    def advance_time(self, seconds):
        beats = self.get_beat_wait_from_time_wait(seconds)
        self.advance(beats)
        return beats, seconds

    def go_to_beat(self, b):
        self._beats = b
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

    def show_plot(self, title=None, resolution=25, show_segment_divisions=True, units="tempo",
                  x_range=None, y_range=None):
        # if we're past the end of the envelope, we want to plot that as a final constant segment
        # bring_up_to_date adds that segment, but we don't want to modify the original envelope, so we deepcopy
        env_to_plot = deepcopy(self).bring_up_to_date() if self.end_time() < self.beats() else self

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
