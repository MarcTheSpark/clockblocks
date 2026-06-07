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

import math
from typing import Union, Sequence, Tuple, TYPE_CHECKING
from cb2.utilities import current_clock
from cb2.moment import Moment
from cb2.enums import DurationUnits
if TYPE_CHECKING:
    from cb2.clock import Clock


class MetricPhaseTarget:

    """
    Class representing a particular point in a (beat or measure) cycle. Implements the ResolvableMoment
    protocol, so a MetricPhaseTarget can be passed anywhere a Moment can: wait / wait_until /
    fork / schedule_action.

    :param phase_or_phases: Where we are in the cycle.
    :param divisor: Length of the cycle (defaults to one beat, meaning that this specifies where we are in the beat)
    :param relative: Whether or not the start of the cycle is measured relative to the current beat/time or to the
        start of the clock.
    :param units: "beats" (default) or "time" — whether the phase and divisor are measured in the clock's beats or
        in its time. (Note: `units` governs resolve() only. Consumers that already choose an axis from their own
        context (e.g. certain methods of TempoEnvelope) ignore this attribute.
    """

    def __init__(self, phase_or_phases: Union[float, Sequence[float]], divisor: float = 1, relative: bool = False,
                 units: str | DurationUnits = "beats"):
        self.phases = (phase_or_phases, ) if not hasattr(phase_or_phases, "__len__") else phase_or_phases
        if not all(0 <= x < divisor for x in self.phases):
            raise ValueError("One or more phases out of range for divisor.")
        self.divisor = divisor
        self.relative = relative
        self.units = DurationUnits(units)

    @classmethod
    def interpret(cls, value: float| Sequence) -> 'MetricPhaseTarget':
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

    def resolve(self, clock: 'Clock') -> Moment:
        """
        Resolve this phase target to an absolute Moment on `clock`: the nearest *future* moment in the clock
        (either a beat or time, depending on self.units) whose metric phase matches. Satisfies the ResolvableMoment
        protocol.
        """
        # get_nearest_matching_* returns the nearest match below and above, in order of nearness;
        # we want the match above, since it's in the future, so we use max.
        if self.units == DurationUnits.BEATS:
            return Moment.at_beat(max(*self.get_nearest_matching_beats(clock.beat())))
        return Moment.at_time(max(*self.get_nearest_matching_times(clock.time())))

    def __repr__(self):
        return "MetricPhaseTarget({}{}{}{})".format(
            str(self.phases[0]) if hasattr(self.phases, "__len__") else self.phases,
            (", " + str(self.divisor)) if self.divisor != 1 else "",
            ", True" if self.relative else "",
            f", units='{self.units.value}'" if self.units != DurationUnits.BEATS else "",
        )
