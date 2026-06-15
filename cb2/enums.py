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

from enum import StrEnum


class DurationUnits(StrEnum):
    """
    Units with which we measure duration or Moment positioning. Either musical BEATS, or the integrated
    TIME taken by those beats at the clock's tempo.

    Note that time is only true seconds on the master clock.
    Within a clock tree, a child's TIME is the same as its parent's BEATS, and is affected by the rates of
    all clocks above it in the tree. Use :meth:`~cb2.clock.Clock.time_in_master` for true seconds.

    A ``StrEnum``, so the bare strings "beats" / "time" are accepted interchangeably."""
    BEATS = "beats"
    TIME = "time"

    @property
    def opposite(self) -> 'DurationUnits':
        """The other axis: ``BEATS.opposite`` is ``TIME`` and vice versa. Used to get the *free* axis
        (the one a `when` did not pin) when solving alignment."""
        return DurationUnits.TIME if self is DurationUnits.BEATS else DurationUnits.BEATS


class TempoUnits(StrEnum):
    """
    The three different, and mutually determined, ways of expressing tempo.

    - TEMPO is the standard BPM understood by musicians
    - RATE is in beats per second, and useful for reasoning about tempo relationships in a clock tree
    - BEATLENGTH is the duration of one beat; mostly useful internally as the unit we're integrating over

    A ``StrEnum``, so the bare strings are accepted.
    """
    TEMPO = "tempo"
    RATE = "rate"
    BEATLENGTH = "beatlength"
