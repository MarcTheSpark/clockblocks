"""
Suite-wide time compression.

Set the environment variable ``CLOCKBLOCKS_TEST_COMPRESSION`` to run the whole suite on a compressed clock, e.g.::

    CLOCKBLOCKS_TEST_COMPRESSION=10 python -m unittest discover -s tests -t .

The default (unset, or ``1``) is ordinary real time, so behavior is unchanged. When a factor is set,
:func:`install` overrides ``clockblocks.scheduler._default_time_backend`` so *every* Clock/Scheduler the suite
constructs (that doesn't pass an explicit backend) runs on a :class:`~clockblocks.scheduler.CompressedTime` of that
factor — no per-test construction change needed.

Tests must then reason in the **scheduler's** time domain, since real wall time runs ``factor``x faster:

  * measure elapsed with :func:`elapsed` (real elapsed * FACTOR -> scheduler-seconds),
  * realize off-clock "fall behind" delay with :func:`sleep` (sleeps ``scheduler_seconds / FACTOR`` real),

so the assertions are identical at any factor. Note handshake/context-switch overhead is *not* compressed,
so large factors loosen achievable tolerances (see ``README.md``); 10x is a good default.
"""
import os
import time

FACTOR = float(os.environ.get("CLOCKBLOCKS_TEST_COMPRESSION", "1"))


def stopwatch() -> float:
    """Start a scheduler-domain stopwatch; pass the result to :func:`elapsed`."""
    return time.perf_counter()


def elapsed(start: float) -> float:
    """Scheduler-domain seconds since ``start`` (real elapsed scaled up by FACTOR)."""
    return (time.perf_counter() - start) * FACTOR


def sleep(scheduler_seconds: float) -> None:
    """Block for ``scheduler_seconds`` of scheduler-time (i.e. ``/ FACTOR`` real seconds)."""
    time.sleep(scheduler_seconds / FACTOR)


def install() -> None:
    """If a compression factor is set, point the scheduler's default-backend factory at a CompressedTime."""
    if FACTOR != 1:
        from clockblocks import scheduler
        from clockblocks.scheduler import CompressedTime
        scheduler._default_time_backend = lambda: CompressedTime(FACTOR)
