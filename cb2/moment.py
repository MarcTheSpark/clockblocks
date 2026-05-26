from __future__ import annotations
from typing import Protocol, runtime_checkable, TYPE_CHECKING
from cb2.enums import DurationUnits
if TYPE_CHECKING:
    from cb2.clock import Clock


@runtime_checkable
class ResolvableMoment(Protocol):
    """
    Anything that can name a point on a clock's timeline. The scheduling layer (wait / wait_until /
    fork / schedule_action) accepts any ResolvableMoment and calls resolve(clock) to pin it to an
    absolute Moment on that clock. Moment and MetricPhaseTarget both implement this.
    """
    def resolve(self, clock: Clock) -> Moment:
        ...


class Moment:
    """
    Represents a point on a clock's timeline. Can be either in beats or in time (as defined by `units`), and can
    be either relative (measured from a clock's current beat/time) or absolute (measured from the clock's start).

    User code should generally construct via one of the classmethods:
        Moment.at_beat(8)       # absolute: beat 8 of the clock
        Moment.at_time(8)       # absolute: 8 seconds (in the clock's time) since it started
        Moment.after_beats(2)   # relative: 2 beats from now
        Moment.after_time(0.5)  # relative: 0.5 seconds from now

    resolve(clock) returns an *absolute* Moment on that clock (a no-op if already absolute). The final scheduling
    process works only with absolute Moments, which are either pinned to a specific beat or a specific time in
    the clock's timeline. Relative moments or MetricPhaseTargets are resolved at scheduling time into absolute
    moments.
    """

    def __init__(self, value: float, units: str | DurationUnits = "beats", relative: bool = False):
        self.value = value
        self.units = DurationUnits(units)
        self.relative = relative

    @classmethod
    def at_beat(cls, beat: float) -> Moment:
        return cls(beat, DurationUnits.BEATS, relative=False)

    @classmethod
    def at_time(cls, time: float) -> Moment:
        return cls(time, DurationUnits.TIME, relative=False)

    @classmethod
    def after_beats(cls, beats: float) -> Moment:
        return cls(beats, DurationUnits.BEATS, relative=True)

    @classmethod
    def after_time(cls, time: float) -> Moment:
        return cls(time, DurationUnits.TIME, relative=True)

    def resolve(self, clock: Clock) -> Moment:
        if not self.relative:
            return self
        now = clock.beat() if self.units == DurationUnits.BEATS else clock.time()
        return Moment(now + self.value, self.units, relative=False)

    def beat_on(self, clock: Clock) -> float:
        """This (absolute) moment expressed as a beat of `clock`."""
        if self.relative:
            raise ValueError("beat_on() requires an absolute Moment; call resolve(clock) first.")
        return self.value if self.units == DurationUnits.BEATS else clock.tempo_history.beat_at_time(self.value)

    def scheduler_time(self, clock: Clock) -> float:
        """The scheduler-time at which this (absolute) moment occurs on `clock`."""
        if self.relative:
            raise ValueError("scheduler_time() requires an absolute Moment; call resolve(clock) first.")
        return clock.clock_to_scheduler_time(self.value, units=self.units)

    def __repr__(self):
        if self.relative:
            name = "after_beats" if self.units == DurationUnits.BEATS else "after_time"
        else:
            name = "at_beat" if self.units == DurationUnits.BEATS else "at_time"
        return f"Moment.{name}({self.value})"


def to_absolute_moment(when: float | ResolvableMoment | None, clock: Clock, *,
                       units_if_number: str | DurationUnits = DurationUnits.BEATS,
                       relative_if_number: bool = True) -> Moment:
    """
    Coerce a `when` argument into an absolute Moment on `clock`. A bare number is wrapped per the
    caller's convention (`units_if_number` / `relative_if_number`); None is treated as "now" (relative 0);
    anything else is assumed to be a ResolvableMoment and resolved.
    """
    if when is None:
        when = 0
    if isinstance(when, (int, float)):
        return Moment(when, units_if_number, relative=relative_if_number).resolve(clock)
    return when.resolve(clock)
