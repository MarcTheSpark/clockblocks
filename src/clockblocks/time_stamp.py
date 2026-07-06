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

from __future__ import annotations
from functools import total_ordering
from typing import TYPE_CHECKING
from clockblocks.utilities import current_clock, snap_float_to_nice_decimal

if TYPE_CHECKING:
    from clockblocks.clock import Clock


@total_ordering
class TimeStamp:
    """
    A clock-agnostic point on the scheduler's timeline, projectable into any clock's beat or time axis
    on demand. Usually captured for the current moment with :meth:`now` (e.g. at note-on); later queries
    answer "what beat / time was this in clock X?" for any clock in the same family.

    :param scheduler_time: the scheduler-time (canonical, not relative to any clock) this TimeStamp
        is pinned to. Per-clock beats and times are later derived from it.
    :param master_clock: the master clock of the clock family this TimeStamp belongs to.
    :ivar scheduler_time: the scheduler-time this TimeStamp represents.
    """

    # Implementation: we store only the scheduler-time and the master. Each query runs that scheduler-time
    # back through the target clock's tempo history (Clock.scheduler_to_clock_time). Tempo histories only
    # grow past the committed point, so resolution is cheap and needs no cached per-clock beat snapshots
    # (as we did in the original clockblocks).
    def __init__(self, scheduler_time: float, master_clock: 'Clock'):
        self.scheduler_time = scheduler_time
        self._master = master_clock

    @classmethod
    def now(cls, master_clock: 'Clock' = None) -> 'TimeStamp':
        """
        Capture a TimeStamp for the current moment. Any clock in the family may be given (its master is
        used); if omitted, the clock is taken from the current thread.
        """
        from clockblocks.clock import Clock
        clock = current_clock() if master_clock is None else master_clock
        if not isinstance(clock, Clock):
            raise ValueError("No valid clock given or found for TimeStamp")
        return cls(clock.scheduler.time(), clock.master)

    @property
    def time_in_master(self) -> float:
        """The time (in the master clock's own time frame) at which this TimeStamp was created."""
        return self._master.scheduler_to_clock_time(self.scheduler_time, desired_units="time")

    def beat_in_clock(self, clock: 'Clock') -> float:
        """Get the beat in ``clock`` at the moment this TimeStamp represents."""
        if clock.master is not self._master:
            raise ValueError("Clock is not in the same family as this TimeStamp")
        return clock.scheduler_to_clock_time(self.scheduler_time, desired_units="beats")

    def time_in_clock(self, clock: 'Clock') -> float:
        """Get the time in ``clock`` at the moment this TimeStamp represents."""
        if clock.master is not self._master:
            raise ValueError("Clock is not in the same family as this TimeStamp")
        return clock.scheduler_to_clock_time(self.scheduler_time, desired_units="time")

    def __sub__(self, other: 'TimeStamp') -> 'TimeStampInterval':
        """``end_stamp - start_stamp`` gives the oriented interval between the two moments."""
        if not isinstance(other, TimeStamp):
            return NotImplemented
        if other._master is not self._master:
            raise ValueError("Cannot subtract TimeStamps from different clock families")
        return TimeStampInterval(other.scheduler_time, self.scheduler_time, self._master)

    def __repr__(self):
        return f"TimeStamp[{self.scheduler_time}]"

    def __eq__(self, other):
        if not isinstance(other, TimeStamp):
            return NotImplemented
        return self.scheduler_time == other.scheduler_time

    def __lt__(self, other):
        if not isinstance(other, TimeStamp):
            return NotImplemented
        return self.scheduler_time < other.scheduler_time

    def __hash__(self):
        return hash(self.scheduler_time)


class TimeStampInterval:
    """
    The oriented span between two moments on the scheduler's timeline, produced by subtracting two TimeStamps
    (``end_stamp - start_stamp``). Like :class:`TimeStamp` it stores only scheduler times (and the associated
    master clock) and resolves on demand.

    Note that this interval is an anchored interval, not a free-floating duration. Its start and end points
    are particular moments on the scheduler, and the beat duration of that span in a given clock depends on
    that clock's absolute tempo curve between those end points.

    Note that, due to floating point arithmetic, two time stamps that were created with a clean time delta
    may yield a slightly noisy delta when the difference is taken (``(start + 0.1) - start`` is not exactly
    ``0.1`` in IEEE-754). For this reason, the duration methods here snap their results to a nice decimal if
    one is nearby, restoring any intended round durations.

    :ivar start_scheduler_time: scheduler-time of the earlier endpoint.
    :ivar end_scheduler_time: scheduler-time of the later endpoint.
    """

    def __init__(self, start_scheduler_time: float, end_scheduler_time: float, master_clock: 'Clock'):
        self.start_scheduler_time = start_scheduler_time
        self.end_scheduler_time = end_scheduler_time
        self._master = master_clock

    @property
    def time_duration_in_master(self) -> float:
        """How much time elapsed across this interval, in the master clock's own time frame."""
        return snap_float_to_nice_decimal(
            self._master.scheduler_to_clock_time(self.end_scheduler_time, desired_units="time")
            - self._master.scheduler_to_clock_time(self.start_scheduler_time, desired_units="time")
        )

    def beats_duration_in_clock(self, clock: 'Clock') -> float:
        """How many beats elapsed in ``clock`` across this interval."""
        if clock.master is not self._master:
            raise ValueError("Clock is not in the same family as this TimeStampInterval")
        return snap_float_to_nice_decimal(
            clock.scheduler_to_clock_time(self.end_scheduler_time, desired_units="beats")
            - clock.scheduler_to_clock_time(self.start_scheduler_time, desired_units="beats")
        )

    def time_duration_in_clock(self, clock: 'Clock') -> float:
        """How much time elapsed across this interval, in ``clock``'s own time frame."""
        if clock.master is not self._master:
            raise ValueError("Clock is not in the same family as this TimeStampInterval")
        return snap_float_to_nice_decimal(
            clock.scheduler_to_clock_time(self.end_scheduler_time, desired_units="time")
            - clock.scheduler_to_clock_time(self.start_scheduler_time, desired_units="time")
        )

    def __repr__(self):
        return f"TimeStampInterval[{self.start_scheduler_time} -> {self.end_scheduler_time}]"