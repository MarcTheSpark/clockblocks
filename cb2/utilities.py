from __future__ import annotations
import math
import threading
import time
from typing import TYPE_CHECKING, Callable, Sequence, Union
if TYPE_CHECKING:
    from cb2 import clock, metric_phase, moment


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


def snap_float_to_nice_decimal(x: float, order_of_magnitude_difference=7) -> float:
    """
    If x is near to a nice decimal, this rounds it. E.g., given a number like 8.01399999999999214, we want to round
    it to 8.014. We do this by comparing what we get if we round coarsely to what we get if we round precisely,
    where order_of_magnitude_difference represents how much more precise the precise round is than the course round.
    If they're the same, then we should be rounding.

    :param x: number to snap
    :param order_of_magnitude_difference: how many orders of magnitude we compare rounding across
    :return: the rounded value
    """
    for first_place in range(0, 17 - order_of_magnitude_difference):
        if round(x, first_place) == round(x, first_place + order_of_magnitude_difference):
            return round(x, first_place)
    return x


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


class _UnsynchronizedSentinel:
    """Marker bound (as a thread's ``__clock__``) to a thread spawned by fork_unsynchronized. Such a
    thread has no clock — current_clock() is None and it can't fork child clocks — but it is allowed
    to use the sleep-based wait()/wait_forever(), which become plain real-time sleeps."""
    def __repr__(self):
        return "<unsynchronized>"


_UNSYNCHRONIZED = _UnsynchronizedSentinel()


def _thread_clock_attr():
    """The raw ``__clock__`` tag on the current thread: a Clock, the _UNSYNCHRONIZED sentinel, or
    None (an ordinary thread that never entered the clock system). wait() needs the sentinel/None
    distinction, which current_clock() flattens away."""
    return getattr(threading.current_thread(), '__clock__', None)


def current_clock() -> clock.Clock | None:
    # utility for getting the clock we are currently using (we attach it to the thread when it's started)
    c = _thread_clock_attr()
    return None if c is _UNSYNCHRONIZED else c


def _spawn_unsynchronized(forked_function: Callable, args: Sequence, kwargs: dict) -> None:
    """Start `forked_function` on a new daemon thread tagged as unsynchronized, so it may use the
    sleep-based waits (current_clock() stays None there). Backs Clock.fork_unsynchronized."""
    kwargs = {} if kwargs is None else kwargs

    def runner():
        threading.current_thread().__clock__ = _UNSYNCHRONIZED
        forked_function(*args, **kwargs)

    threading.Thread(target=runner, daemon=True).start()


def wait(dt: float, units="beats") -> None:
    c = _thread_clock_attr()
    if c is _UNSYNCHRONIZED:
        time.sleep(dt)  # no clock => no tempo; units is ignored (dt is real seconds)
    elif c is not None:
        c.wait(dt, units=units)
    else:
        from cb2.clock import NoActiveClockError
        raise NoActiveClockError("wait() called on a thread with no active clock.")


def wait_forever() -> None:
    """
    Block forever on the currently active clock (see :meth:`Clock.wait_forever`) — usually to keep the
    main script alive while child clocks do the work. On an unsynchronized thread, sleeps indefinitely;
    on an ordinary (non-clock) thread, raises NoActiveClockError.
    """
    c = _thread_clock_attr()
    if c is _UNSYNCHRONIZED:
        while True:
            time.sleep(1)
    elif c is not None:
        c.wait_forever()
    else:
        from cb2.clock import NoActiveClockError
        raise NoActiveClockError("wait_forever() called on a thread with no active clock.")


def wait_for_children_to_finish() -> None:
    """
    Block on the currently active clock until its child clocks have finished
    (see :meth:`Clock.wait_for_children_to_finish`). Requires a real clock (an unsynchronized or
    ordinary thread has no children) — raises NoActiveClockError otherwise.
    """
    c = current_clock()
    if c is None:
        from cb2.clock import NoActiveClockError
        raise NoActiveClockError("wait_for_children_to_finish() called on a thread with no active clock.")
    c.wait_for_children_to_finish()


def fork_unsynchronized(forked_function: Callable, args: Sequence = (), kwargs: dict = None) -> None:
    """
    Spawn `forked_function` as an asynchronous thread, not on a child clock (see
    :meth:`Clock.fork_unsynchronized`). If there is no active clock on this thread, falls back to a
    plain ``threading.Thread``.
    """
    c = current_clock()
    if c is None:
        _spawn_unsynchronized(forked_function, args, kwargs or {})
    else:
        c.fork_unsynchronized(forked_function, args=args, kwargs=kwargs)


def fork(forked_function: Callable, args: Sequence = (), kwargs: dict = None, name: str = None,
         initial_rate: float = None, initial_tempo: float = None, initial_beat_length: float = None,
         when: Union[float, 'moment.ResolvableMoment'] = None,
         done_callback: Callable = None) -> 'clock.Clock':
    """
    Fork `forked_function` as a child of the currently active clock (see :meth:`Clock.fork`).
    Requires a real clock to fork from — raises NoActiveClockError otherwise.
    """
    c = current_clock()
    if c is None:
        from cb2.clock import NoActiveClockError
        raise NoActiveClockError("Cannot fork: there is no active clock on this thread.")
    return c.fork(forked_function, args=args, kwargs=kwargs, name=name, initial_rate=initial_rate,
                  initial_tempo=initial_tempo, initial_beat_length=initial_beat_length,
                  when=when, done_callback=done_callback)
