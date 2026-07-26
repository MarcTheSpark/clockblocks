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
"""
Module containing the free functions that make up clockblocks' most convenient interface. Rather than calling
methods on a particular clock, these operate on whichever clock is running the current thread: most importantly
:func:`wait` and :func:`fork`, along with getters and setters for that clock's tempo (:func:`set_tempo`,
:func:`set_tempo_target`, :func:`apply_tempo_envelope`, and their rate/beat-length counterparts).
"""

from __future__ import annotations
import math
import threading
from typing import TYPE_CHECKING, Callable, Sequence, Union
from clockblocks.exceptions import NoActiveClockError
if TYPE_CHECKING:
    from clockblocks import clock, moment
    from clockblocks.moment import ResolvableMoment
    from clockblocks.tempo_envelope import TempoEnvelope


# Default tolerances for near-equality comparisons of beats / times.
# Because of the use of numerical integration, two values representing "the same moment"
# can easily differ by ~1e-12. 1e-9 sits far below any musically meaningful gap, making
# it a safe threshold while still distinguishing genuine (audible) differences.
NEAR_EQUAL_REL_TOL = 1e-9
NEAR_EQUAL_ABS_TOL = 1e-12


def near_equal(a: float, b: float, *, rel_tol: float = NEAR_EQUAL_REL_TOL,
               abs_tol: float = NEAR_EQUAL_ABS_TOL) -> bool:
    """
    True if ``a`` and ``b`` are equal to within floating-point reconstruction noise.

    Thin wrapper over :func:`math.isclose` with project-wide default tolerances plus a small absolute
    floor: ``math.isclose`` defaults ``abs_tol=0.0``, which is too strict near zero, and several
    callers compare quantities that can be ~0 (e.g. envelope-segment durations).
    """
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def meaningfully_less_than(a: float, b: float, *, rel_tol: float = NEAR_EQUAL_REL_TOL,
                           abs_tol: float = NEAR_EQUAL_ABS_TOL) -> bool:
    """
    True if ``a`` is meaningfully less than ``b`` — i.e. ``a < b`` and not merely by rounding
    noise (see :func:`near_equal`). Use in place of a bare ``<`` when ``a`` and ``b`` are beats/times
    reconstructed from scheduler time.
    """
    return a < b and not near_equal(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def meaningfully_greater_than(a: float, b: float, *, rel_tol: float = NEAR_EQUAL_REL_TOL,
                              abs_tol: float = NEAR_EQUAL_ABS_TOL) -> bool:
    """
    True if ``a`` is meaningfully greater than ``b`` (mirror of :func:`meaningfully_less_than`).
    """
    return a > b and not near_equal(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


# snap_float_to_nice_decimal now lives in expenvelope (the base Envelope needs it for durations rounding);
# re-exported here so existing `from clockblocks.utilities import snap_float_to_nice_decimal` importers keep working.
from expenvelope import snap_float_to_nice_decimal


class _PrintColors:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def current_clock() -> clock.Clock | None:
    """
    Get the :class:`~clockblocks.clock.Clock` active on the current thread, or None if none is active.
    """
    # The clock is attached to its thread (as __clock__) when the thread is started.
    return getattr(threading.current_thread(), '__clock__', None)


##################################################################################################################
#                                  Context-inferring wait and fork methods
##################################################################################################################
# Module-level forwarders that grab the clock active on the calling thread via current_clock() and
# forward to the corresponding Clock method.
# ---------------------------------------------------------------------------------------------------------------


def wait(dt: 'float | ResolvableMoment', units="beats") -> None:
    """
    Block the clock currently active on this thread for ``dt`` beats (or seconds, if ``units="time"``),
    yielding to the scheduler. Forwards to :meth:`~clockblocks.clock.Clock.wait`.

    ``dt`` may also be a :class:`~clockblocks.moment.Moment` or
    :class:`~clockblocks.metric_phase.MetricPhaseTarget`, in which case it is resolved directly and ``units`` is
    ignored — e.g. ``wait(Moment.at_beat(8))``.

    On a thread with no active clock, raises NoActiveClockError.

    :param dt: how long to wait — a number (in beats, or seconds if ``units="time"``), or a Moment /
        MetricPhaseTarget to wait until.
    :param units: either ``"beats"`` or ``"time"`` (ignored when ``dt`` is a Moment).
    """
    c = current_clock()
    if c is not None:
        c.wait(dt, units=units)
    else:
        raise NoActiveClockError("wait() called on a thread with no active clock.")


def wait_until(when: 'float | ResolvableMoment', units="beats") -> None:
    """
    Block the clock currently active on this thread until the *absolute* beat (or time, if ``units="time"``)
    given by ``when``. Forwards to :meth:`~clockblocks.clock.Clock.wait_until`.

    ``when`` may also be a :class:`~clockblocks.moment.Moment` or
    :class:`~clockblocks.metric_phase.MetricPhaseTarget`, in which case the units are ignored and behavior is
    identical to wait(). If ``when`` is already in the past, this returns essentially immediately.

    On a thread with no active clock, raises NoActiveClockError.

    :param when: the absolute beat (or time, if ``units="time"``) to wait until, or a Moment /
        MetricPhaseTarget to resolve.
    :param units: either ``"beats"`` or ``"time"`` (ignored when ``when`` is a Moment).
    """
    c = current_clock()
    if c is not None:
        c.wait_until(when, units=units)
    else:
        raise NoActiveClockError("wait_until() called on a thread with no active clock.")


def wait_forever() -> None:
    """
    Block forever on the currently active clock (see :meth:`~clockblocks.clock.Clock.wait_forever`) — usually
    to keep the main script alive while child clocks do the work. Unblocks only if the clock is killed, raising
    :class:`~clockblocks.exceptions.ClockKilledError`. On a thread with no active clock, raises NoActiveClockError.
    """
    c = current_clock()
    if c is not None:
        c.wait_forever()
    else:
        raise NoActiveClockError("wait_forever() called on a thread with no active clock.")


def wait_for_children_to_finish() -> None:
    """
    Block on the currently active clock until its child clocks have finished
    (see :meth:`~clockblocks.clock.Clock.wait_for_children_to_finish`). Requires a real clock —
    raises NoActiveClockError otherwise.
    """
    c = current_clock()
    if c is None:
        raise NoActiveClockError("wait_for_children_to_finish() called on a thread with no active clock.")
    c.wait_for_children_to_finish()


def fork(forked_function: Callable, args: Sequence = (), kwargs: dict = None, name: str = None,
         initial_rate: float = None, initial_tempo: float = None, initial_beat_length: float = None,
         when: Union[float, 'moment.ResolvableMoment'] = None,
         done_callback: Callable = None) -> 'clock.Clock':
    """
    Run ``forked_function`` on a new child clock of the currently active clock, so it proceeds in
    parallel while staying coordinated under the same musical time. See :meth:`~clockblocks.clock.Clock.fork`;
    raises NoActiveClockError if there is no active clock.

    :param forked_function: the function to run on the child clock. Note that, unlike the original clockblocks,
        the child clock is *not* injected as an argument — call :func:`current_clock` from inside the function
        if you need a reference to it.
    :param args: positional arguments passed to ``forked_function``.
    :param kwargs: keyword arguments passed to ``forked_function``.
    :param name: optional name for the child clock, useful for debugging.
    :param initial_rate: the child's starting rate in beats per second. Give at most one of
        ``initial_rate``/``initial_tempo``/``initial_beat_length``; if none is given the child starts at
        rate 1.
    :param initial_tempo: the child's starting tempo in beats per minute.
    :param initial_beat_length: the child's starting beat length in seconds per beat.
    :param when: when to start the fork. ``None`` (the default) starts it immediately; otherwise pass a
        :class:`~clockblocks.moment.Moment` (e.g. ``Moment.after_beats(4)`` or ``Moment.at_time(10)``) or a
        :class:`~clockblocks.metric_phase.MetricPhaseTarget`. Unlike :func:`wait`, a bare number is rejected
        here, since we don't know from context whether it's beats/time, relative/absolute.
    :param done_callback: an optional function called when the forked function finishes.
    :return: the newly created child :class:`~clockblocks.clock.Clock`.
    """
    c = current_clock()
    if c is None:
        raise NoActiveClockError("Cannot fork: there is no active clock on this thread.")
    return c.fork(forked_function, args=args, kwargs=kwargs, name=name, initial_rate=initial_rate,
                  initial_tempo=initial_tempo, initial_beat_length=initial_beat_length,
                  when=when, done_callback=done_callback)


##################################################################################################################
#                                      Context-inferring tempo methods
##################################################################################################################
# Module-level forwarders that grab the clock active on the calling thread via current_clock() and
# forward to the corresponding Clock tempo modifier. Note that these act on the current clock and *not*
# on the master clock. set_tempo(120) inside a forked clock changes *that fork's* tempo; to change the
# master's tempo, call the method on the master Clock object directly.
# ---------------------------------------------------------------------------------------------------------------

def _current_clock_or_raise(caller: str) -> 'clock.Clock':
    """
    Returns the clock active on this thread, or raises NoActiveClockError naming `caller`.
    (Unlike wait(), there is no unsynchronized fallback: an unsynchronized thread has no tempo to change.)
    """
    c = current_clock()
    if c is None:
        raise NoActiveClockError(f"{caller}() called on a thread with no active clock.")
    return c


def set_tempo(tempo: float) -> None:
    """Immediately set the tempo of the current clock (see :attr:`~clockblocks.clock.Clock.tempo`). Raises
    NoActiveClockError if there is no active clock.

    :param tempo: the new tempo, in beats per minute."""
    _current_clock_or_raise("set_tempo").tempo = tempo


def set_rate(rate: float) -> None:
    """Immediately set the rate of the current clock (see :attr:`~clockblocks.clock.Clock.rate`). Raises
    NoActiveClockError if there is no active clock.

    :param rate: the new rate, in beats per second."""
    _current_clock_or_raise("set_rate").rate = rate


def set_beat_length(beat_length: float) -> None:
    """Immediately set the beat length of the current clock (see :attr:`~clockblocks.clock.Clock.beat_length`).
    Raises NoActiveClockError if there is no active clock.

    :param beat_length: the new beat length, in seconds per beat."""
    _current_clock_or_raise("set_beat_length").beat_length = beat_length


def get_tempo() -> float:
    """Return the current tempo (in beats per minute) of the currently active clock (see
    :attr:`~clockblocks.clock.Clock.tempo`). Raises NoActiveClockError if there is no active clock.

    :return: the current tempo, in beats per minute."""
    return _current_clock_or_raise("get_tempo").tempo


def get_rate() -> float:
    """Return the current rate (in beats per second) of the currently active clock (see
    :attr:`~clockblocks.clock.Clock.rate`). Raises NoActiveClockError if there is no active clock.

    :return: the current rate, in beats per second."""
    return _current_clock_or_raise("get_rate").rate


def get_beat_length() -> float:
    """Return the current beat length (in seconds per beat) of the currently active clock (see
    :attr:`~clockblocks.clock.Clock.beat_length`). Raises NoActiveClockError if there is no active clock.

    :return: the current beat length, in seconds per beat."""
    return _current_clock_or_raise("get_beat_length").beat_length


##################################################################################################################
#                                     Context-inferring position readers
##################################################################################################################
# Module-level readers of the current clock's position, for user convenience (current_clock().beat is a bit wordy),
# and consistency with other context-inferring readers. Note that there are deliberately no
# projected_beat()/projected_time() counterparts, since these are intended for callers *outside* the clock family.
# ---------------------------------------------------------------------------------------------------------------


def get_beat() -> float:
    """Return how many beats have passed on the currently active clock (see :attr:`~clockblocks.clock.Clock.beat`).

    As with the tempo helpers, this reads *this* clock and not the master; use ``current_clock().master.beat`` for
    that. Raises NoActiveClockError if there is no active clock.

    :return: the current beat of the active clock."""
    # float() sheds the _CallableFloat that Clock.beat returns, so this new function doesn't inherit the
    # deprecated beat() call spelling. Drop the cast when that shim goes in 2.0.
    return float(_current_clock_or_raise("get_beat").beat)


def get_time() -> float:
    """Return how much time has passed on the currently active clock (see :attr:`~clockblocks.clock.Clock.time`) —
    in seconds if it is the master clock, or parent clock beats if it is a forked child clock.

    As with the tempo helpers, this reads *this* clock and not the master; use ``current_clock().master.time`` for
    that. Raises NoActiveClockError if there is no active clock.

    :return: the current time of the active clock."""
    # float() as in get_beat: sheds the 1.x-only _CallableFloat shim.
    return float(_current_clock_or_raise("get_time").time)


def set_tempo_target(tempo_target: float, when: 'ResolvableMoment', curve_shape: float = None,
                     truncate: bool = True, align_to: 'ResolvableMoment' = None) -> None:
    """Smoothly change the current clock's tempo to ``tempo_target`` (in beats per minute), arriving at
    ``when``. Forwards to :meth:`~clockblocks.clock.Clock.set_tempo_target` on the currently active clock. Raises
    NoActiveClockError if there is no active clock.

    :param tempo_target: the tempo to arrive at, in beats per minute.
    :param when: when the target is reached, as a :class:`~clockblocks.moment.Moment` (e.g. ``Moment.after_beats(4)``,
        ``Moment.at_time(10)``) or a :class:`~clockblocks.metric_phase.MetricPhaseTarget`.
    :param curve_shape: the bend of the transition: ``0`` is linear, ``> 0`` changes late, ``< 0`` early.
    :param truncate: if ``True``, discard any tempo curve already scheduled past the current beat first.
    :param align_to: optional; solve the curvature to land the free axis (the one ``when`` did not pin)
        on a phase or coordinate — e.g. accelerate over a fixed time, landing on a downbeat. See
        :meth:`~clockblocks.clock.Clock.set_tempo_target`."""
    _current_clock_or_raise("set_tempo_target").set_tempo_target(
        tempo_target, when, curve_shape=curve_shape, truncate=truncate, align_to=align_to)


def set_rate_target(rate_target: float, when: 'ResolvableMoment', curve_shape: float = None,
                    truncate: bool = True, align_to: 'ResolvableMoment' = None) -> None:
    """Smoothly change the current clock's rate to ``rate_target`` (in beats per second), arriving at
    ``when``. Forwards to :meth:`~clockblocks.clock.Clock.set_rate_target` on the currently active clock. Raises
    NoActiveClockError if there is no active clock.

    :param rate_target: the rate to arrive at, in beats per second.
    :param when: when the target is reached, as a :class:`~clockblocks.moment.Moment` or
        :class:`~clockblocks.metric_phase.MetricPhaseTarget`.
    :param curve_shape: the bend of the transition: ``0`` is linear, ``> 0`` changes late, ``< 0`` early.
    :param truncate: if ``True``, discard any tempo curve already scheduled past the current beat first.
    :param align_to: optional; solve the curvature to land the free axis on a phase or coordinate. See
        :meth:`~clockblocks.clock.Clock.set_rate_target`."""
    _current_clock_or_raise("set_rate_target").set_rate_target(
        rate_target, when, curve_shape=curve_shape, truncate=truncate, align_to=align_to)


def set_beat_length_target(beat_length_target: float, when: 'ResolvableMoment', curve_shape: float = None,
                           truncate: bool = True, align_to: 'ResolvableMoment' = None) -> None:
    """Smoothly change the current clock's beat length to ``beat_length_target`` (in seconds per beat),
    arriving at ``when``. Forwards to :meth:`~clockblocks.clock.Clock.set_beat_length_target` on the currently
    active clock. Raises NoActiveClockError if there is no active clock.

    :param beat_length_target: the beat length to arrive at, in seconds per beat.
    :param when: when the target is reached, as a :class:`~clockblocks.moment.Moment` or
        :class:`~clockblocks.metric_phase.MetricPhaseTarget`.
    :param curve_shape: the bend of the transition: ``0`` is linear, ``> 0`` changes late, ``< 0`` early.
    :param truncate: if ``True``, discard any tempo curve already scheduled past the current beat first.
    :param align_to: optional; solve the curvature to land the free axis on a phase or coordinate. See
        :meth:`~clockblocks.clock.Clock.set_beat_length_target`."""
    _current_clock_or_raise("set_beat_length_target").set_beat_length_target(
        beat_length_target, when, curve_shape=curve_shape, truncate=truncate, align_to=align_to)


def set_tempo_targets(tempo_targets: Sequence[float], whens: 'Sequence[ResolvableMoment]',
                      curve_shapes: Sequence[float] = None, truncate: bool = True,
                      align_to: 'ResolvableMoment | Sequence[ResolvableMoment | None]' = None) -> None:
    """Set several tempo targets at once, building a multi-segment tempo curve on the current clock.
    Forwards to :meth:`~clockblocks.clock.Clock.set_tempo_targets`; acts on the currently active clock. Raises
    NoActiveClockError if there is no active clock.

    :param tempo_targets: the tempo to arrive at for each segment, in beats per minute.
    :param whens: the arrival moment for each target (same length as ``tempo_targets``). Each is resolved
        against the clock's *current* position (so an ``after_*`` moment counts from now, not from the
        previous segment's end), beats and time may be mixed, and the moments must come out strictly
        increasing in clock-time.
    :param curve_shapes: optional per-segment curve shapes (same length), or ``None`` for all-linear.
    :param truncate: if ``True``, discard any tempo curve already scheduled past the current beat first.
    :param align_to: optional; bend whole runs of segments to land collectively on a phase/coordinate.
        Pass a single :class:`~clockblocks.moment.Moment`/:class:`~clockblocks.metric_phase.MetricPhaseTarget` to
        align the whole call at its end, or a per-segment list of ``None``/targets where each non-``None`` entry
        closes and aligns the run since the previous alignment. See
        :meth:`~clockblocks.clock.Clock.set_beat_length_targets`.

    This is one-shot. To loop a tempo shape, build a :class:`~clockblocks.tempo_envelope.TempoEnvelope` and pass it to
    :func:`apply_tempo_envelope` with ``loop=True``."""
    _current_clock_or_raise("set_tempo_targets").set_tempo_targets(
        tempo_targets, whens, curve_shapes=curve_shapes, truncate=truncate, align_to=align_to)


def set_rate_targets(rate_targets: Sequence[float], whens: 'Sequence[ResolvableMoment]',
                     curve_shapes: Sequence[float] = None, truncate: bool = True,
                     align_to: 'ResolvableMoment | Sequence[ResolvableMoment | None]' = None) -> None:
    """Set several rate targets at once, building a multi-segment tempo curve on the current clock.
    Forwards to :meth:`~clockblocks.clock.Clock.set_rate_targets`; acts on the currently active clock. Raises
    NoActiveClockError if there is no active clock.

    :param rate_targets: the rate to arrive at for each segment, in beats per second.
    :param whens: the arrival moment for each target (same length as ``rate_targets``).
    :param curve_shapes: optional per-segment curve shapes (same length), or ``None`` for all-linear.
    :param truncate: if ``True``, discard any tempo curve already scheduled past the current beat first.
    :param align_to: optional; align whole runs of segments to a phase/coordinate. See
        :meth:`~clockblocks.clock.Clock.set_beat_length_targets`."""
    _current_clock_or_raise("set_rate_targets").set_rate_targets(
        rate_targets, whens, curve_shapes=curve_shapes, truncate=truncate, align_to=align_to)


def set_beat_length_targets(beat_length_targets: Sequence[float], whens: 'Sequence[ResolvableMoment]',
                            curve_shapes: Sequence[float] = None, truncate: bool = True,
                            align_to: 'ResolvableMoment | Sequence[ResolvableMoment | None]' = None) -> None:
    """Set several beat-length targets at once, building a multi-segment tempo curve on the current clock.
    Forwards to :meth:`~clockblocks.clock.Clock.set_beat_length_targets`; acts on the currently active clock. Raises
    NoActiveClockError if there is no active clock.

    :param beat_length_targets: the beat length to arrive at for each segment, in seconds per beat.
    :param whens: the arrival moment for each target (same length as ``beat_length_targets``).
    :param curve_shapes: optional per-segment curve shapes (same length), or ``None`` for all-linear.
    :param truncate: if ``True``, discard any tempo curve already scheduled past the current beat first.
    :param align_to: optional; align whole runs of segments to a phase/coordinate. See
        :meth:`~clockblocks.clock.Clock.set_beat_length_targets`."""
    _current_clock_or_raise("set_beat_length_targets").set_beat_length_targets(
        beat_length_targets, whens, curve_shapes=curve_shapes, truncate=truncate, align_to=align_to)


def apply_tempo_function(function: Callable, domain_start: float = 0, domain_end: float = None,
                         duration_units: str = "beats", truncate: bool = True, loop: bool = False,
                         extension_increment: float = 2.0, **kwargs) -> None:
    """Drive the current clock's tempo (in beats per minute) by following a function of beats/time.
    Forwards to :meth:`~clockblocks.clock.Clock.apply_tempo_function`; acts on the currently active clock. Raises
    NoActiveClockError if there is no active clock.

    :param function: a callable mapping a position (in the units given by ``duration_units``) to a tempo.
    :param domain_start: the input value at which to start reading ``function``.
    :param domain_end: the input value at which to stop, or ``None`` to follow the function open-endedly.
    :param duration_units: ``"beats"`` or ``"time"`` — whether the function's input is measured in beats
        or in seconds.
    :param truncate: if ``True``, discard any tempo curve already scheduled past the current beat first.
    :param loop: only relevant when ``domain_end`` is set: if ``True``, repeat that finite domain
        indefinitely (until :func:`stop_tempo_loop_or_function`).
    :param extension_increment: only relevant when ``domain_end`` is ``None``: how far ahead (in the
        function's domain) to project at a time, extending again as the clock reaches the projected end.
    :param kwargs: forwarded to the underlying envelope-from-function sampling
        (``scanning_step_size``, ``iterations``, etc. — see
        :meth:`~clockblocks.tempo_envelope.TempoHistory.apply_function`)."""
    _current_clock_or_raise("apply_tempo_function").apply_tempo_function(
        function, domain_start=domain_start, domain_end=domain_end, duration_units=duration_units,
        truncate=truncate, loop=loop, extension_increment=extension_increment, **kwargs)


def apply_rate_function(function: Callable, domain_start: float = 0, domain_end: float = None,
                        duration_units: str = "beats", truncate: bool = True, loop: bool = False,
                        extension_increment: float = 2.0, **kwargs) -> None:
    """Drive the current clock's rate (in beats per second) by following a function of beats/time.
    Forwards to :meth:`~clockblocks.clock.Clock.apply_rate_function`; acts on the currently active clock.
    Raises NoActiveClockError if there is no active clock. See :func:`apply_tempo_function` for the meaning
    of every argument (the only difference is that ``function`` returns a rate in beats per second)."""
    _current_clock_or_raise("apply_rate_function").apply_rate_function(
        function, domain_start=domain_start, domain_end=domain_end, duration_units=duration_units,
        truncate=truncate, loop=loop, extension_increment=extension_increment, **kwargs)


def apply_beat_length_function(function: Callable, domain_start: float = 0, domain_end: float = None,
                               duration_units: str = "beats", truncate: bool = True, loop: bool = False,
                               extension_increment: float = 2.0, **kwargs) -> None:
    """Drive the current clock's beat length (in seconds per beat) by following a function of beats/time.
    Forwards to :meth:`~clockblocks.clock.Clock.apply_beat_length_function`; acts on the currently active clock.
    Raises NoActiveClockError if there is no active clock. See :func:`apply_tempo_function` for the meaning
    of every argument (the only difference is that ``function`` returns a beat length in seconds per beat)."""
    _current_clock_or_raise("apply_beat_length_function").apply_beat_length_function(
        function, domain_start=domain_start, domain_end=domain_end, duration_units=duration_units,
        truncate=truncate, loop=loop, extension_increment=extension_increment, **kwargs)


def apply_tempo_envelope(envelope: 'TempoEnvelope', truncate: bool = True, loop: bool = False) -> None:
    """Append a ready-made :class:`~clockblocks.tempo_envelope.TempoEnvelope` onto the current clock's tempo
    curve. This is the way to *loop* a tempo shape (a :class:`~clockblocks.tempo_envelope.TempoEnvelope` is
    defined over beats, so it loops unambiguously).
    Forwards to :meth:`~clockblocks.clock.Clock.apply_tempo_envelope`; acts on the currently active clock. Raises
    NoActiveClockError if there is no active clock.

    :param envelope: the tempo envelope to apply (under the hood a beat-length-over-beats curve).
    :param truncate: if ``True``, discard any tempo curve already scheduled past the current beat first.
    :param loop: if ``True``, repeat the envelope indefinitely until stopped with
        :func:`stop_tempo_loop_or_function`."""
    _current_clock_or_raise("apply_tempo_envelope").apply_tempo_envelope(
        envelope, truncate=truncate, loop=loop)


def stop_tempo_loop_or_function() -> None:
    """Stop following any function or looping envelope previously started on the current clock's tempo
    (via :func:`apply_tempo_function`/:func:`apply_rate_function`/:func:`apply_beat_length_function` or a
    looping :func:`apply_tempo_envelope`). The tempo holds at its current value. Forwards to
    :meth:`~clockblocks.clock.Clock.stop_tempo_loop_or_function`; acts on the currently active clock. Raises
    NoActiveClockError if there is no active clock."""
    _current_clock_or_raise("stop_tempo_loop_or_function").stop_tempo_loop_or_function()
