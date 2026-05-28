from __future__ import annotations
from functools import total_ordering
from typing import TYPE_CHECKING
from cb2.utilities import current_clock

if TYPE_CHECKING:
    from cb2.clock import Clock


@total_ordering
class TimeStamp:
    """
    A snapshot of "now" as a clock-agnostic point on the scheduler's timeline, projectable
    into any clock's beat or time frame on demand.

    Constructed at the moment of interest (e.g. note-on); later queries answer "what beat /
    what time was this in clock X?" by running the cached scheduler-time back through
    ``Clock.scheduler_to_clock_time``. Because tempo histories are append-only past the
    committed point, this resolution is cheap and the old master-side ``time_stamp_data``
    dedup cache is no longer needed.

    :param clock: any clock in the family of interest; if None, the clock is captured
        implicitly from the current thread.
    :ivar scheduler_time: the scheduler-time at which this TimeStamp was created. This is
        the canonical, family-invariant axis; per-clock beats/times are derived from it.
    """

    def __init__(self, clock: 'Clock' = None):
        from cb2.clock import Clock
        clock = current_clock() if clock is None else clock
        if not isinstance(clock, Clock):
            raise ValueError("No valid clock given or found for TimeStamp")
        self.scheduler_time = clock.scheduler.time()
        self._master = clock.master

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