"""
Module defining the central :class:`Clock` class — recursively nestable clocks that coordinate musical
time under a master clock and its background :class:`~clockblocks.scheduler.Scheduler`.
"""

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

import functools
import logging
import math
import sys
import threading
import traceback
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from itertools import count
from typing import Callable, Sequence, Iterator
from copy import deepcopy
from clockblocks.tempo_envelope import TempoEnvelope, TempoHistory
from clockblocks.scheduler import Scheduler, TimingBackend
from clockblocks.moment import Moment, ResolvableMoment, to_absolute_moment
from clockblocks.metric_phase import MetricPhaseTarget
from clockblocks.enums import DurationUnits
from clockblocks.utilities import _PrintColors, current_clock
from clockblocks.exceptions import (ClockKilledError, DeadClockError, WrongThreadError,
                            NotMasterClockError)
import textwrap


class ClockState(Enum):
    PENDING = "pending"  # forked, awaiting start_delay; thread not yet running user code
    ALIVE = "alive"      # forked_function is running (or about to run)
    DEAD = "dead"        # killed, or forked_function returned


@dataclass(frozen=True)
class ClockFamilyOptions:
    """
    Fine-tuning options for a clock family, passed to a master :class:`Clock` on construction.
    A master and all its descendants share one scheduler and one thread pool; these knobs configure both.

    The live knobs (``timing_policy``, ``precise_timing``, ``spin_guard_duration``) can be changed after
    construction via the matching :class:`Clock` properties; the thread pool knobs are construction-only
    because the thread pool is built once.

    :param timing_policy:
        0.0 = absolute timing (measure from clock start and catch up as much as possible),
        1.0 = relative timing (wait the full delay each time, even when behind),
        floats in between clamp how far a wait may be compressed/stretched from its nominal length to
        stay on the absolute schedule. Default value is 0.98 (near-relative: a wait may shave up to 2%
        of its length per event to catch up).
    :param precise_timing:
        When true, use a busy-wait in the immediate run-up to a scheduled action to arrive as precisely
        as possible. (OS wait is by nature jittery.) Costs one core for at most ``spin_guard_duration``
        seconds per event. Helps under any ``timing_policy`` by tightening whatever that policy optimizes.
    :param spin_guard_duration:
        Width (seconds) of the busy-spin guard band used when ``precise_timing`` is on. Default 500µs.
    :param pool_size:
        Max worker threads in the family's shared ThreadPoolExecutor, used for every call to ``fork``.
        Workers are created lazily up to this cap; past it, a forked clock falls back to a plain thread
        and warns that the pool has run out of threads.
    :param prewarm_pool:
        How many pool workers to spin up eagerly at construction, front-loading the (sub-millisecond)
        thread-creation cost of the first that-many forks. Clamped to ``pool_size``; 0 disables, leaving
        an idle clock with no threads at all.
    """
    timing_policy: float = 0.98
    precise_timing: bool = False
    spin_guard_duration: float = 0.0005
    pool_size: int = 200
    prewarm_pool: int = 10
    time_backend: 'TimingBackend | None' = None  # None -> real perf_counter time; tests pass CompressedTime

    def __post_init__(self):
        if not 0.0 <= self.timing_policy <= 1.0:
            raise ValueError("timing_policy must be between 0 (absolute) and 1 (relative).")
        if self.spin_guard_duration < 0:
            raise ValueError("spin_guard_duration must be non-negative.")
        if self.pool_size < 1:
            raise ValueError("pool_size must be at least 1.")
        if self.prewarm_pool < 0:
            raise ValueError("prewarm_pool must be non-negative.")


# ======================================================================================================
# Clock termination / lifecycle paths
# ------------------------------------------------------------------------------------------------------
# Every clock ends in one of four ways. They converge on the same final state (DEAD, detached from the
# parent, scheduler released), but reach it differently. kill() and _fork_wrapper both implement parts of
# this; this is the single reference for how they fit together.
#
#   SUB-CLOCK, natural end:
#       The user function returns inside _fork_wrapper (no exception). _fork_wrapper removes the child
#       from parent._children, sets state=DEAD, then notifies _scheduler_park_condition (the scheduler is
#       parked there awaiting a next wait() that will never come). The thread exits.
#
#   SUB-CLOCK, killed:
#       kill() (possibly cascaded from an ancestor) sets state=DEAD, removes the clock's queued wake-up
#       (and any other event for it) from the scheduler heap, sets _wait_event, notifies
#       _scheduler_park_condition, and detaches it from its parent. If it was parked inside wait(), it
#       wakes, sees DEAD, and raises ClockKilledError. If it wasn't in wait(), its next wait() raises
#       DeadClockError at the entry check. Either error propagates into _fork_wrapper, which catches it and
#       falls through to the same cleanup as the natural-end path (detach + notify are idempotent).
#
#   MASTER, natural end:
#       The owning thread finishes its top-level code. There is no _fork_wrapper, so nothing notifies the
#       scheduler — it stays parked on the master's _scheduler_park_condition. On the main thread this is
#       harmless (the thread ending is process exit, and the daemon scheduler dies with it). On another
#       thread in a long-lived process it leaks the parked scheduler daemon until exit, which is why a master
#       created off the main thread should be killed explicitly (see Clock's class docstring).
#
#   MASTER, killed:
#       kill() sets state=DEAD, removes queued wake-ups, sets _wait_event, and notifies
#       _scheduler_park_condition (releasing the scheduler — critical here, since there's no _fork_wrapper
#       to do it later). Finally, because a master owns its scheduler 1:1, killing the master also kills
#       that scheduler so its thread doesn't park forever. Two sub-cases depending on who calls kill():
#         - Killed from another thread (e.g. a GUI/pygame callback) while the owning thread is parked in
#           a wait: owning thread wakes, sees DEAD, and raises ClockKilledError out of the wait. Every wait
#           variant propagates it (wait_forever / wait_for_children_to_finish included — they no longer
#           swallow), so it's caught at the managing boundary: run_as_server's server loop or a
#           `with clock:` block for a master, or the user's own try/except.
#         - Self-kill (the owning thread calls master.kill() itself): it can only do so while running, not
#           parked, so nothing observes DEAD here — kill() just winds things down and returns normally. No
#           ClockKilledError; a *later* wait()/fork() on this thread would raise DeadClockError at entry.
# ======================================================================================================


def _reschedule_after_tempo_change(fn):
    """
    Decorator for Clock tempo/rate/beat_length setters. Brings the target clock's tempo_history up to
    the current scheduler position (so the change applies forward, not retroactively from the last
    committed beat), applies the mutation, then recomputes the scheduler-time of any queued wakeups
    whose `acting_clock` is self or a descendant.

    Two concurrency hazards, two locks:
      * The clock tree must not change while we enumerate descendants to reschedule them, so we hold
        `_tree_lock` throughout.
      * A clock mutates its own tempo_history in wait()'s cleanup, as well as possibly from user code,
        so we must not rewrite tempo_history from an outside thread while a clock is awake. This is not an issue
        for any of the other threads *within* the clock system, since only one clock can be awake at a time.
        On the other hand, when a tempo-modifier is called from a thread *external to the clock system*
        (current_clock() is None), we must wait for a dormant window where no clocks are executing. We do
        this via `scheduler.while_quiescent()`. Note that, if we tried to use while_quiescent() from a clock
        thread it would deadlock (see its docstring).

    The lock order — `_execution_lock` (via while_quiescent) then `_tree_lock` — matters: while_quiescent()
    blocks until the scheduler finishes its current action, and the clock running that action may itself
    need `_tree_lock` (to fork, kill, or finish and detach). Holding `_tree_lock` while waiting in
    while_quiescent() would therefore deadlock.
    """
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        def apply():
            self.bring_up_to_date()
            result = fn(self, *args, **kwargs)
            self._reschedule_self_and_descendants()
            return result

        if current_clock() is None:
            # External thread: wait until no clock is awake, then mutate.
            with self.scheduler.while_quiescent(), self._tree_lock:
                return apply()
        # On a clock's own thread: scheduler is frozen on us; only the tree needs guarding.
        with self._tree_lock:
            return apply()
    return wrapper


def _threadpool_error_callback(e: BaseException) -> None:
    # Reports exceptions raised inside a pool task. A Future stashes its exception and never re-raises it
    # unless someone reads .result()/.exception() (we don't), so without this they vanish silently. We
    # surface them here in red on stderr, mimicking the traceback an uncaught thread exception would print.
    exc_type = type(e).__name__
    formatted_traceback = ''.join(traceback.format_tb(e.__traceback__))
    print(f"{_PrintColors.RED}Error encountered on forked clock. "
          f"Traceback (most recent call last):\n{formatted_traceback}{exc_type}: {e}{_PrintColors.END}",
          file=sys.stderr)


class Clock:
    """
    Recursively nestable clock. Clocks can fork child-clocks, which can in turn fork their own
    child-clocks. A clock with no parent is the *master*, and the whole family stays coordinated under it.

    A master created on the main thread needs no cleanup. If you create one on another thread, call
    ``master.kill()`` when you're done with it so its background timing thread doesn't linger (or use
    :meth:`run_as_server`, which manages that thread for you).

    :param name: (optional) can be useful for keeping track in confusing multi-threaded situations
    :param parent: the parent clock for this clock; a value of None indicates the master clock
    :param initial_rate: starting rate of this clock (if set, don't set initial tempo or beat length)
    :param initial_tempo: starting tempo of this clock (if set, don't set initial rate or beat length)
    :param initial_beat_length: starting beat length of this clock (if set, don't set initial tempo or rate)
    :param clock_family_options: (master-only) a :class:`ClockFamilyOptions` bundling the family-level
        timing/threading knobs (``timing_policy``, ``precise_timing``, ``spin_guard_duration``,
        ``pool_size``, ``prewarm_pool``). The live knobs are also adjustable afterwards via
        :attr:`timing_policy`, :attr:`precise_timing`, and :attr:`spin_guard_duration`. Passing it
        to a forked child raises an error, since a child shares its master's scheduler and pool.
    :ivar name: the name of this clock (string)
    :ivar parent: the parent Clock to which this clock belongs (Clock, or None if master clock)
    :ivar tempo_history: TempoHistory describing how this clock has changed or will change tempo
    """

    def __init__(self, name: str | None= None, parent: Clock | None = None, initial_rate: float | None = None,
                 initial_tempo: float | None = None, initial_beat_length: float | None = None,
                 clock_family_options: ClockFamilyOptions | None = None):
        if clock_family_options is not None and parent is not None:
            raise NotMasterClockError(
                "clock_family_options is master-only: a forked child shares its master's scheduler and pool. "
                "Set these on the master, or adjust the live knobs via the timing_policy / precise_timing / "
                "spin_guard_duration properties.")
        options = clock_family_options if clock_family_options is not None else ClockFamilyOptions()
        self.name = name
        self.parent = parent
        self._children = []
        # Counter handed out to *this* clock's future children; their clock_id suffix
        # comes from this counter so siblings get distinct, monotonically-increasing ids.
        self._child_counter = count()
        self.clock_id = (0,) if parent is None else parent.clock_id + (next(parent._child_counter),)
        # Scheduler-event priority: deeper clocks sort *before* shallower ones at the same scheduled
        # time, so when a parent and its descendant both wake at t, the descendant fires first.
        # Among sibling clocks, the one forked first fires first. 
        # So we sort first by reverse depth (negative length of id) and then by the id which represents forking order
        # One quirk: if you have two cousin clocks, one forked earlier from a later-forked parent, and one forked
        # later from an earlier-forked parent, the one with the earlier parent wins. Not sure if this matters much
        self._priority = (-len(self.clock_id), self.clock_id)

        # tempo envelope, in seconds since I was created
        self.tempo_history = TempoHistory(
            Clock._rate_tempo_or_beat_length_to_rate(initial_rate, initial_tempo, initial_beat_length),
            units="rate"
        )

        # A clock family shares exactly one scheduler, created along with the master clock.
        # Children inherit the master scheduler from their parent. There is deliberately no
        # way to hand a master an existing scheduler: if you want two clocks in sync,
        # fork them from one master; if you don't, make two masters and they run on independent
        # schedulers with independent timing policies.
        if self.parent is not None:
            self.scheduler = self.parent.scheduler
        else:
            self.scheduler = Scheduler(timing_policy=options.timing_policy,
                                       precise_timing=options.precise_timing,
                                       spin_guard_duration=options.spin_guard_duration, daemon=True,
                                       time_backend=options.time_backend)
            self.scheduler.start()

        self._scheduler_park_condition = threading.Condition()
        self._wait_event = threading.Event()
        # Set while this clock is parked in wait_for_children_to_finish(). It scheduled no wakeup of its
        # own (an indefinite _wait(None)), so _detach_child watches this flag and, when the last child
        # detaches, schedules the wakeup that releases it. See _detach_child / wait_for_children_to_finish.
        self._waiting_for_children = False

        if self.is_master():
            # The whole family shares one thread pool, owned by the master, that carries out every fork
            # in the family (see _run_in_pool). Reusing pooled workers is significantly cheaper
            # than spawning a fresh Thread per fork (which is important for rapid note playback in SCAMP, since
            # each note playback is done with a fork). That said, benchmarking suggests the actual time-cost of
            # raw thread creation is pretty small; it's not clear if this is worth it.
            #
            # We use concurrent.futures.ThreadPoolExecutor rather than multiprocessing.pool.ThreadPool on
            # purpose: the latter builds an internal multiprocessing SimpleQueue whose two SemLocks are real
            # OS semaphores, which on macOS (spawn start method) is reported by the resource_tracker as "2 leaked
            # semaphore objects to clean up at shutdown" whenever the process is Ctrl-C'd before they're
            # unlinked. ThreadPoolExecutor is pure-threading (no multiprocessing, no semaphores) and spawns
            # its workers lazily up to max_workers, so an idle Session costs no threads.
            #
            # _pool_semaphore tracks how many active threads we are using, since if we go over max_workers
            # new _run_in_pool would block until one is free. By tracking we can switch to a raw thread
            # when we reach the capacity of the ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(max_workers=options.pool_size,
                                            thread_name_prefix=f"clock-pool-{self.name or id(self)}")
            self._pool_semaphore = threading.BoundedSemaphore(options.pool_size)
            self._prewarm_pool(min(options.prewarm_pool, options.pool_size))

            # The whole family shares one structural lock, owned by the master. fork / kill / external
            # tempo changes acquire it (via _tree_lock) so the clock tree (_children, _state) and any
            # enumeration of it stays invariant for the duration of a structural operation.
            self._clock_tree_lock = threading.RLock()
            # Master starts ALIVE (no fork / start_delay)
            self._state = ClockState.ALIVE
            # the first thing we do is stop and put things in the scheduler's hands
            # tell the scheduler to wake up right away and get this clock going, then wait for the scheduler to do it
            self.scheduler.schedule_action(self.scheduler.time(), self._wake_and_advance_to_next_wait_call,
                                           self._priority,
                                           {"description": f"Initial wake for Clock(name={self.name!r})",
                                            "acting_clock": self})
            self._wait_event.wait()

            threading.current_thread().__clock__ = self
            # the "parent" of the master clock, timing wise, is the scheduler. But master_clock.parent is None
            # still because there is not parent clock.
            self.parent_offset = self._start_time_in_scheduler = self.scheduler.time()
        else:
            # Forked children start PENDING and flip to ALIVE at the top of _fork_wrapper
            # once their start_delay has elapsed.
            self._state = ClockState.PENDING
            self.parent_offset = self.parent.beat()
            self._start_time_in_scheduler = self.parent_offset + self.parent._start_time_in_scheduler

    @staticmethod
    def _rate_tempo_or_beat_length_to_rate(rate, tempo, beat_length) -> float:
        if rate is tempo is beat_length is None:
            return 1
        if not (rate is None) + (tempo is None) + (beat_length is None) == 2:
            # exactly one of the arguments must be non-None
            raise ValueError("No more than one of `rate`, `tempo`, or `beat_length` may be defined.")
        return rate if rate is not None else 1 / beat_length if beat_length is not None else tempo / 60

    ##################################################################################################################
    #                                                 Family Matters
    ##################################################################################################################

    @property
    def master(self) -> 'Clock':
        """
        The master clock under which this clock operates (possibly itself)
        """
        return self if self.is_master() else self.parent.master

    @property
    def _tree_lock(self) -> threading.RLock:
        """
        The single per-family RLock guarding clock-tree structure (`_children` / `_state`) and any
        enumeration of it. Lives on the master; fork, kill, and external tempo changes acquire it so
        two structural operations can't interleave (e.g. a fork's create+append racing a kill's
        collect+remove, which would otherwise orphan the new child). Lock order, when combined with the
        scheduler's locks, is `_execution_lock -> _tree_lock -> _queue_change_condition`.
        """
        return self.master._clock_tree_lock

    def is_master(self) -> bool:
        """
        Check if this is the master clock

        :return: True if this is the master clock, False otherwise
        """
        return self.parent is None

    def while_scheduler_quiescent(self):
        """
        Context manager that ensures the scheduler is not mid-action while the body runs. Wraps
        :meth:`Scheduler.while_quiescent` with a same-family skip: if the calling thread is already a
        clock thread of this scheduler (either a clock executing user code between waits, or a foreign
        thread already inside an outer ``while_scheduler_quiescent`` block that tagged ``__clock__``),
        re-acquiring the lock would deadlock — so we yield a no-op instead, since the caller already
        has exclusive access by construction.

        Use this anywhere foreign-thread callers might mutate state a scheduled action reads
        (e.g. a pygame handler calling scamp's :meth:`~scamp.transcriber.Transcriber.start_transcribing`).
        """
        active_clock = getattr(threading.current_thread(), '__clock__', None)
        if getattr(active_clock, 'scheduler', None) is self.scheduler:
            return nullcontext()
        return self.scheduler.while_quiescent()

    def children(self) -> Sequence['Clock']:
        """
        Get all direct child clocks forked by this clock.

        :return: tuple of all child clocks of this clock
        """
        return tuple(self._children)

    def _detach_child(self, child: 'Clock') -> None:
        """
        Remove `child` from this clock's child list. The caller must hold `_tree_lock` (both call sites —
        _fork_wrapper's natural-exit cleanup and kill()'s teardown — already do).

        If this clock is alive and parked in wait_for_children_to_finish(), and `child` was the last one,
        release it by scheduling an immediate _wake_and_advance_to_next_wait_call on the scheduler. This
        is the same wake mechanism as a normally scheduled wake up; it's just that in this case it is triggered
        by observing all children have finished and scheduled immediately.
        
        We go through the scheduler (rather than just setting _wait_event) so the parent resumes in the normal
        post-wait state — woken with the scheduler re-parked on its own _scheduler_park_condition — and is
        free to wait() again. The detaching child is mid-cleanup with the scheduler parked on *its*
        condition; once it releases the scheduler, that immediate event is the next thing to run.
        """
        if child in self._children:
            self._children.remove(child)
        if self._waiting_for_children and not self._children and self._state is ClockState.ALIVE:
            self._waiting_for_children = False
            self.scheduler.schedule_action(self.scheduler.time(), self._wake_and_advance_to_next_wait_call,
                                           self._priority,
                                           {"description": f"{self} wake (children finished)",
                                            "acting_clock": self})

    def iterate_inheritance(self, include_self: bool = True) -> Iterator['Clock']:
        """
        Iterate through parent, grandparent, etc. of this clock up until the master clock

        :param include_self: whether or not to include this clock in the iterator or start with the parent
        :return: iterator going up the clock family tree up to the master clock
        """

        clock = self
        if include_self:
            yield clock
        while clock.parent is not None:
            clock = clock.parent
            yield clock

    def inheritance(self, include_self: bool = True) -> Sequence['Clock']:
        """
        Get all parent, grandparent, etc. of this clock up until the master clock

        :param include_self: whether or not to include this clock in the iterator or start with the parent
        :return: tuple containing the clock's inheritance
        """
        return tuple(self.iterate_inheritance(include_self))

    def iterate_all_relatives(self, include_self: bool = False) -> Iterator['Clock']:
        """
        Iterate through all related clocks to this clock.

        :param include_self: whether or not to include this clock in the iterator
        :return: iterator going through all clocks in the family tree, starting with the master
        """
        if include_self:
            return self.master.iterate_descendants(True)
        else:
            return (c for c in self.master.iterate_descendants(True) if c is not self)

    def iterate_descendants(self, include_self: bool = False) -> Iterator['Clock']:
        """
        Iterate through all children, grandchildren, etc. of this clock

        :param include_self: whether or not to include this clock in the iterator
        :return: iterator going through all descendants
        """
        if include_self:
            yield self
        for child_clock in self._children:
            yield child_clock
            for descendant_of_child in child_clock.iterate_descendants():
                yield descendant_of_child

    def descendants(self) -> Sequence['Clock']:
        """
        Get all children, grandchildren, etc. of this clock

        :return: tuple of all descendants
        """
        return tuple(self.iterate_descendants())

    def print_family_tree(self) -> None:
        """
        Print a hierarchical representation of this clock's family tree.
        """
        print(self.master._child_tree_string(self))

    def _child_tree_string(self, highlight_clock: 'Clock' = None) -> str:
        name_text = self.name if self.name is not None else "(UNNAMED)"
        if highlight_clock is self:
            name_text = _PrintColors.BOLD + name_text + _PrintColors.END
        children = self.children()
        if len(children) == 0:
            return name_text
        return "{}:\n{}".format(
            name_text,
            textwrap.indent("\n".join(child._child_tree_string(highlight_clock) for child in self.children()), "  ")
        )

    ##################################################################################################################
    #                                        Boilerplate TempoHistory Functionality
    ##################################################################################################################

    def time(self, projected: bool = False) -> float:
        """
        How much time has passed since this clock was created. Either in seconds, if this is the master
        clock, or in beats in the parent clock, if this clock was the result of a call to fork.

        Computed on demand from the scheduler's committed position, so it is consistent from any thread
        without the owning thread having had to advance its `tempo_history` pointer first. Note, however,
        that the scheduler's position is **event-quantized, not wall-clock-interpolated**: it is bumped to
        each event's time as that event executes (across the whole family), so a clock reading its own
        position while its action runs sees the exact current value, but a read taken *between* events
        (from a non-clock thread) is frozen at the time of the most recent event.

        :param projected: if ``True``, return a wall-clock-interpolated estimate of the current position
            (via :meth:`Scheduler.projected_time`) instead of the committed, event-quantized value. Useful
            for a smoothly-advancing read from a non-clock thread between events; the default ``False`` gives
            the committed value that everything else in the family agrees on.
        :return: the elapsed time (see units above).

        Does not mutate `tempo_history` (the committed pointer).
        """
        scheduler_time = self.scheduler.projected_time() if projected else self.scheduler.time()
        return self.scheduler_to_clock_time(scheduler_time, desired_units="time")

    def beat(self, projected: bool = False) -> float:
        """
        How many beats have passed since this clock was created. See :meth:`time` for thread/laziness
        semantics and for the ``projected`` flag.

        :param projected: if ``True``, return a wall-clock-interpolated estimate of the current beat rather
            than the committed, event-quantized value. See :meth:`time`.
        :return: the elapsed beats.
        """
        scheduler_time = self.scheduler.projected_time() if projected else self.scheduler.time()
        return self.scheduler_to_clock_time(scheduler_time, desired_units="beats")

    def time_in_master(self) -> float:
        """
        This clock's current position projected onto the master's time axis (true seconds since the
        master was created). Equivalent to ``self.master.time()``, since master.time() is itself live
        and any thread reads the same scheduler-derived value.
        """
        return self.master.time()

    def wall_time_in_scheduler(self) -> float:
        """
        How long has this clock been alive in the scheduler.
        """
        return self.scheduler.wall_time() - self._start_time_in_scheduler

    def status(self, verbose: bool = False) -> str:
        """
        A snapshot of this clock's name, beat, time, and wall time in the scheduler.
        """
        name = self.name if self.name is not None else "UNNAMED"
        if verbose:
            return (f"Clock {name!r}\n"
                    f"  beat: {self.beat():.9f}\n"
                    f"  time: {self.time():.9f}\n"
                    f"  wall: {self.wall_time_in_scheduler():.9f}")
        return f"[{name} beat={self.beat():.3f} time={self.time():.3f} wall={self.wall_time_in_scheduler():.3f}]"

    def print_status(self, verbose: bool = False) -> None:
        """Print this clock's :meth:`status` snapshot."""
        print(self.status(verbose), flush=True)

    @property
    def beat_length(self) -> float:
        """
        The length of a beat in this clock in seconds.
        Note that beat_length, tempo and rate are interconnected properties, and that by setting one of them the
        other two are automatically set in response according to the relationship: beat_length = 1/rate = 60/tempo.
        Also, note that "seconds" refers to actual seconds only in the master clock; otherwise it refers to beats
        in the parent clock.
        """
        return self.tempo_history.beat_length

    @beat_length.setter
    @_reschedule_after_tempo_change
    def beat_length(self, b):
        self.tempo_history.beat_length = b

    @property
    def rate(self) -> float:
        """
        The rate of this clock in beats / second.
        Note that beat_length, tempo and rate are interconnected properties, and that by setting one of them the
        other two are automatically set in response according to the relationship: beat_length = 1/rate = 60/tempo.
        Also, note that "seconds" refers to actual seconds only in the master clock; otherwise it refers to beats
        in the parent clock.
        """
        return self.tempo_history.rate

    @rate.setter
    @_reschedule_after_tempo_change
    def rate(self, r):
        self.tempo_history.rate = r

    @property
    def tempo(self) -> float:
        """
        The rate of this clock in beats / minute
        Note that beat_length, tempo and rate are interconnected properties, and that by setting one of them the
        other two are automatically set in response according to the relationship: beat_length = 1/rate = 60/tempo.
        Also, note that "seconds" refers to actual seconds only in the master clock; otherwise it refers to beats
        in the parent clock.
        """
        return self.tempo_history.tempo

    @tempo.setter
    @_reschedule_after_tempo_change
    def tempo(self, t):
        self.tempo_history.tempo = t

    def absolute_rate(self) -> float:
        """
        Rate of this clock in beats / (true) second, with all parent rates folded in.
        """
        return self.rate if self.parent is None else self.rate * self.parent.absolute_rate()

    def absolute_tempo(self) -> float:
        """Tempo (BPM) in true minutes, with all parent rates folded in."""
        return self.absolute_rate() * 60

    def absolute_beat_length(self) -> float:
        """Beat length in true seconds, with all parent rates folded in."""
        return 1 / self.absolute_rate()

    ##################################################################################################################
    #                                          Tempo Targets / Functions
    ##################################################################################################################
    # Thin Clock-level bridges over TempoHistory's tempo-curve mutation API: they wrap the underlying
    # call in `@_reschedule_after_tempo_change` (so queued descendant wake-ups re-project against the
    # new curve and so the mutation acquires `_tree_lock` / `while_quiescent` as appropriate) and
    # express *when* a target is reached as a ResolvableMoment — a `Moment` (after_beats/after_time/
    # at_beat/at_time) or a `MetricPhaseTarget` (which lands the target on the next matching metric
    # phase). The Moment carries its own beats-vs-time axis, so there is no separate `duration_units`,
    # and a MetricPhaseTarget passed as `when` subsumes the old `metric_phase_target` argument.
    # ------------------------------------------------------------------

    def _resolve_when(self, when: 'ResolvableMoment') -> 'tuple[float, DurationUnits]':
        """
        Resolve a `when` (a Moment or MetricPhaseTarget; bare number rejected) into the
        ``(duration, axis)`` pair TempoHistory's setters want, where ``duration`` is the offset from the
        clock's current position and ``axis`` is the ``DurationUnits`` it is measured along
        (``DurationUnits.BEATS`` or ``.TIME`` — a StrEnum, so it compares equal to "beats"/"time", but
        callers rely on it being the enum, e.g. ``axis.opposite``).
        Note that, although a ResolvableMoment often carries the axis information already,
        it is inferred from context when unset, and therefore needs to be returned here as
        part of the process of resolving the `when`.
        """
        abs_when = to_absolute_moment(when, self, allow_number=False)
        pinned_axis = abs_when.units
        # The reference "now" is read live, which is safe even when called repeatedly in a loop (see
        # :meth:`_apply_targets`): ``beat()``/``time()`` derive from the scheduler's committed ``_ideal_time``,
        # which only advances when the scheduler executes an event — and the scheduler is parked/quiescent for
        # the duration of a tempo-setting call, so the reference does not drift between iterations.
        now = self.beat() if pinned_axis == DurationUnits.BEATS else self.time()
        return abs_when.value - now, pinned_axis

    def _resolve_align_to(self, align_to: 'ResolvableMoment | None', pinned_axis: DurationUnits,
                          curve_shape: float | None = None,
                          warn_on_fixed_curve_shape: bool = True) -> 'float | MetricPhaseTarget | None':
        """Resolve an `align_to` (a ResolvableMoment, or None) against the axis `when` pinned, into the
        `alignment_target` TempoHistory / :meth:`_align_run` want — one of:

          * ``None`` — no alignment;
          * a :class:`MetricPhaseTarget` on the *free* axis (the one opposite `pinned_axis`), passed through
            unchanged (its own ``units`` is validated against / inferred as the free axis);
          * a fixed numeric coordinate on the free axis (resolved from a plain Moment).

        A plain Moment or MetricPhaseTarget on the *pinned* axis is an error.

        `curve_shape` is *not* transformed or returned — it is consulted only to warn: with a fixed Moment
        `align_to` the curvature is fully determined, so an explicitly-set curve_shape is meaningless, and we
        warn that it's ignored. (Callers handle curve_shape themselves: the singular setters pass it to the
        segment they build, where the solver overwrites it to hit the target; the group path bakes the
        per-segment curve_shapes in at build time as seeds for :meth:`_align_run`.) `warn_on_fixed_curve_shape`
        gates that warning — the group path passes False for multi-segment runs, where the curvature is *not*
        uniquely determined and per-segment curve_shapes do legitimately seed the distribution."""
        if align_to is None:
            return None

        free_axis = pinned_axis.opposite
        if isinstance(align_to, MetricPhaseTarget):
            if align_to.units is not None and align_to.units != free_axis:
                raise ValueError(
                    f"align_to is fixed to the {align_to.units.value} axis, but `when` already pins that "
                    f"axis; align_to must be on the free ({free_axis.value}) axis (leave its units as None "
                    f"to infer it).")
            # pass the MetricPhaseTarget through as-is; the axis is the free axis, so its own units are ignored
            return align_to

        # a plain Moment represents a fixed coordinate on the free axis
        # we start by resolving it from a possibly relative Moment to an absolute Moment
        abs_align_to = to_absolute_moment(align_to, self, allow_number=False)
        if abs_align_to.units != free_axis:
            raise ValueError(
                f"align_to is on the {abs_align_to.units.value} axis, but `when` pins that axis; align_to "
                f"must be on the free ({free_axis.value}) axis.")

        # for a fixed alignment target, curvature is fully determined, so an explicitly given curve shape
        # will be ignored, and probably represents a misunderstanding. Warn the user (unless this is a
        # multi-segment group run, where curve_shapes do meaningfully seed the distribution)
        if warn_on_fixed_curve_shape and curve_shape:
            warnings.warn("curve_shape is ignored when align_to is a fixed point, as this fully determines "
                          "the required curvature.")
        return abs_align_to.value

    def _apply_targets(self, history_target_setter: Callable, targets: Sequence[float],
                       whens: 'Sequence[ResolvableMoment]', curve_shapes: Sequence[float],
                       truncate: bool, align_to: 'ResolvableMoment | Sequence | None' = None) -> None:
        """Build a multi-segment tempo curve using the given tempo history target setter (`set_beat_length_target`/
        `set_rate_target`/`set_tempo_target`). The levels are given by `targets`, the moments where those levels
        are reached are given by `whens`, which can target either the beats or time axis. The `align_to` argument,
        if provided, allows us to target particular `Moment`s or `MetricPhaseTarget`s on the other axis by bending
        segment curve shapes. It can be any of:

          * ``None``, meaning no alignment. Curve shapes pass through directly.
          * a per-segment list/tuple (same length as `targets`) of ``None``/``ResolvableMoment``. Each non-``None``
            entry causes all segments since the last non-``None` entry to bend collectively so the run lands on
            that target.
          * a single ResolvableMoment, which aligns the *whole* call as one run, landing its end on the given
            alignment target. (equivalent to a per-segment list which is all None except for the final value)

        Note that these when's are resolved relative to the current clock position, *not* the end of the previous
        segment. Any ambiguity resulting from the fact that multiple targets are set at once (possibly mixing beat
        and time axes), is resolved by building the tempo curve incrementally, left-to-right.

        Errors are raised if:

        - The whens do not come out strictly increasing in clock-time (re-raised from the underlying setter)
        - Any of the alignment targets are unreachable
        - A group-aligned run of length > 1 mixes axes. All whens within a run must be on one axis, with the
         alignment target on the other axis.

        The whole build is atomic: any failure restores the pre-call tempo curve.
        """
        # pre-build checks and normalization of targets/curve_shapes/align_to
        num_targets = len(targets)
        if len(whens) != num_targets:
            raise ValueError("Inconsistent number of targets and whens.")
        # None for the whole arg means all-linear; a per-element None also means linear (0), matching the
        # singular setters' curve_shape=None default. Normalizing here is load-bearing, not just tidiness:
        # an un-normalized None reaches _add_segment's time-axis branch and raises (abs(None)).
        curve_shapes = ([0] * num_targets if curve_shapes is None
                        else [0 if cs is None else cs for cs in curve_shapes])
        if len(curve_shapes) != num_targets:
            raise ValueError("Inconsistent number of targets and curve_shapes.")
        # normalize align_to into a per-segment list; a single ResolvableMoment aligns the whole run at its end
        if align_to is None:
            align_to_list = [None] * num_targets
        elif isinstance(align_to, (list, tuple)):
            if len(align_to) != num_targets:
                raise ValueError("Inconsistent number of targets and align_to entries.")
            align_to_list = list(align_to)
        else:
            align_to_list = [None] * (num_targets - 1) + [align_to]

        # call-level snapshot so any failure (a backwards `when`, an unreachable group align) restores the curve.
        backup = deepcopy(self.tempo_history.segments)
        try:
            # loop through targets left-to-right, accumulating the current run. Each `when` resolves against
            # the live clock position, which is stable across the loop (see _resolve_when's docstring).
            run_start = 0          # index of the first segment in the current (not-yet-aligned) run
            run_axes = set()       # the when-axes (DurationUnits) seen so far in the current run
            for i, (target, when, curve_shape, seg_align) in enumerate(
                    zip(targets, whens, curve_shapes, align_to_list)):
                duration, units = self._resolve_when(when)
                run_axes.add(units)
                try:
                    history_target_setter(target, duration, curve_shape=curve_shape, duration_units=units,
                                          truncate=(truncate and i == 0))
                except ValueError as e:
                    now = self.beat() if units == DurationUnits.BEATS else self.time()
                    raise ValueError(
                        f"`when` #{i} ({when!r}) resolves to {units} {duration + now:g}, which does not "
                        f"extend beyond the previous segment; `when`s must be strictly increasing in "
                        f"clock-time."
                    ) from e

                if seg_align is not None:
                    # close the current run [run_start .. i] and bend it collectively onto seg_align
                    run_len = i - run_start + 1
                    if run_len > 1 and len(run_axes) > 1:
                        raise ValueError(
                            f"align_to at index {i} closes a multi-segment run of segments (indices {run_start}–{i}) "
                            f"that mixes beat- and time-axis `when`s; a group-aligned run must operate"
                            f"on a single axis.")
                    pinned_axis = next(iter(run_axes))  # get the shared axis from the set
                    # resolve seg_align onto the free axis.
                    # Re: curve shape, it has already been baked into the newly added segment in the history_setter
                    # call above, and it will be adjusted as needed by _align_run below. The only reason we pass it
                    # to self._resolve_align_to is to warn on a run of length 1 with a fixed target, where an explicit
                    # curve shape was given. (In that case the curve shape is fully determined, and the user-provided
                    # value is ignored, suggesting a misunderstanding on the user's part.)
                    alignment_target = self._resolve_align_to(
                        seg_align, pinned_axis, curve_shape, warn_on_fixed_curve_shape=(run_len == 1))
                    free_axis = pinned_axis.opposite
                    run_segments = self.tempo_history.segments[-run_len:]
                    if not self.tempo_history._align_run(run_segments, alignment_target, free_axis):
                        raise ValueError(
                            f"Could not bend segments {run_start}–{i} to align to {seg_align!r} on the "
                            f"{free_axis.value} axis.")
                    # start a fresh run after the alignment point
                    run_start = i + 1
                    run_axes = set()
        except Exception:
            # restoring the backup reverts the curve without going through a @tempo_modification method, so
            # clear the conversion caches that may hold entries computed against the discarded curve. (Only
            # needed here: in-loop mutations all entry-clear via @tempo_modification, and a successful build
            # leaves the cache consistent with the final curve.)
            self.tempo_history.segments = backup
            self.tempo_history.time_at_beat.cache_clear()
            self.tempo_history.beat_at_time.cache_clear()
            raise

    @_reschedule_after_tempo_change
    def set_beat_length_target(self, beat_length_target: float, when: 'ResolvableMoment',
                               curve_shape: float = None, truncate: bool = True,
                               align_to: 'ResolvableMoment' = None) -> None:
        """Smoothly change this clock's beat length (seconds per beat) to ``beat_length_target``,
        arriving at the moment given by ``when``. This is the underlying representation behind
        :meth:`set_tempo_target` and :meth:`set_rate_target`; a longer beat length means a slower tempo.

        :param beat_length_target: the beat length to arrive at, in seconds per beat.
        :param when: when the target should be reached, as a :class:`Moment` (e.g. ``Moment.after_beats(4)``,
            ``Moment.at_time(10)``) or a :class:`MetricPhaseTarget` (the next point matching a particular
            phase within the beat cycle). ``after_*`` moments count from the clock's *current* position.
            A plain number is rejected; wrap it in a Moment to clarify beat/time and relative/absolute.
        :param curve_shape: the bend of the transition. ``0`` (the default) is a straight line; ``> 0``
            keeps the old tempo longer and changes late; ``< 0`` changes early then eases in. When
            ``align_to`` solves for the curvature (see below) this acts only as a starting hint, and
            with a fixed ``align_to``, it is ignored entirely (and a warning is issued if set).
        :param truncate: if ``True`` (the default), any tempo curve already scheduled past the current
            beat is discarded first, so the change starts from where the clock is right now. If ``False``,
            this target is appended after whatever is already scheduled.
        :param align_to: optional. If `when` is an arrival beat, this specifies the desired arrival time.
            If `when` is an arrival time, this specifies the desired arrival beat. A MetricPhaseTarget can
            be given instead of a singular time/beat, to express when we should arrive within a time/beat
            cycle. Note that the coordination of beat and time is done via mutating the curve shape, so
            an explicit curve shape is only a hint (ignored completely if align_to is a fixed point rather
            than a metric phase target).
        :raises ValueError: if ``when``/``align_to`` share an axis, or if no curvature can reach the
            requested ``align_to`` (in which case the tempo curve is left unchanged)."""
        duration, duration_units = self._resolve_when(when)
        alignment_target = self._resolve_align_to(align_to, duration_units, curve_shape)
        self.tempo_history.set_beat_length_target(
            beat_length_target, duration, curve_shape=(curve_shape or 0), alignment_target=alignment_target,
            duration_units=duration_units, truncate=truncate,
        )

    @_reschedule_after_tempo_change
    def set_rate_target(self, rate_target: float, when: 'ResolvableMoment',
                        curve_shape: float = None, truncate: bool = True,
                        align_to: 'ResolvableMoment' = None) -> None:
        """Smoothly change this clock's rate (beats per second) to ``rate_target``, arriving at ``when``.
        Rate is the reciprocal of beat length: a higher rate is a faster tempo.

        :param rate_target: the rate to arrive at, in beats per second.
        :param when: when the target is reached. See :meth:`set_beat_length_target`.
        :param curve_shape: the bend of the transition. See :meth:`set_beat_length_target`.
        :param truncate: whether to discard already-scheduled tempo changes first. See
            :meth:`set_beat_length_target`.
        :param align_to: optional; solve the curvature to land the free axis on a phase or coordinate.
            See :meth:`set_beat_length_target`.
        :raises ValueError: see :meth:`set_beat_length_target`."""
        duration, duration_units = self._resolve_when(when)
        alignment_target = self._resolve_align_to(align_to, duration_units, curve_shape)
        self.tempo_history.set_rate_target(
            rate_target, duration, curve_shape=(curve_shape or 0), alignment_target=alignment_target,
            duration_units=duration_units, truncate=truncate,
        )

    @_reschedule_after_tempo_change
    def set_tempo_target(self, tempo_target: float, when: 'ResolvableMoment',
                         curve_shape: float = None, truncate: bool = True,
                         align_to: 'ResolvableMoment' = None) -> None:
        """Smoothly change this clock's tempo to ``tempo_target``, arriving at ``when``. Tempo is measured
        in beats per minute; this is usually the most natural of the three equivalent ways to set a target
        (see also :meth:`set_rate_target` and :meth:`set_beat_length_target`).

        :param tempo_target: the tempo to arrive at, in beats per minute (quarter-notes per minute by
            convention).
        :param when: when the target is reached. See :meth:`set_beat_length_target`.
        :param curve_shape: the bend of the transition. See :meth:`set_beat_length_target`.
        :param truncate: whether to discard already-scheduled tempo changes first. See
            :meth:`set_beat_length_target`.
        :param align_to: optional; solve the curvature to land the free axis on a phase or coordinate.
            See :meth:`set_beat_length_target`.
        :raises ValueError: see :meth:`set_beat_length_target`."""
        duration, duration_units = self._resolve_when(when)
        alignment_target = self._resolve_align_to(align_to, duration_units, curve_shape)
        self.tempo_history.set_tempo_target(
            tempo_target, duration, curve_shape=(curve_shape or 0), alignment_target=alignment_target,
            duration_units=duration_units, truncate=truncate,
        )

    @_reschedule_after_tempo_change
    def set_beat_length_targets(self, beat_length_targets: Sequence[float],
                                whens: 'Sequence[ResolvableMoment]',
                                curve_shapes: Sequence[float] = None,
                                truncate: bool = True,
                                align_to: 'ResolvableMoment | Sequence[ResolvableMoment | None]' = None) -> None:
        """Smoothly change this clock's beat length (seconds per beat) through a series of targets,
        building a multi-segment tempo curve in one call — the multi-segment form of
        :meth:`set_beat_length_target`. This is non-looping; to apply a looping tempo shape, build a
        :class:`TempoEnvelope` and use :meth:`apply_tempo_envelope`.

        :param beat_length_targets: the beat lengths (seconds per beat) to arrive at, one per segment.
        :param whens: when each target is reached (same length as ``beat_length_targets``), each a
            :class:`Moment` or :class:`MetricPhaseTarget`. Note that every `when` is resolved against the clock's
            *current* position, so an ``after_*`` moment counts from now, not from the previous segment's.
            Beats- and time-axis moments may be freely mixed across the list, so long as the whens come
            out in strictly increasing clock-time.
        :param curve_shapes: optional per-segment bends (same length as ``beat_length_targets``), each as in
            :meth:`set_beat_length_target`. ``None`` for the whole argument (the default), or a per-element
            ``None``, means linear (``0``) for that segment.
        :param truncate: if ``True`` (the default), any tempo curve already scheduled past the current beat
            is discarded first, so the run starts from where the clock is now; if ``False`` the run is
            appended after whatever is already scheduled.
        :param align_to: optional curvature-solved alignment, extending :meth:`set_beat_length_target`'s
            ``align_to`` to runs of segments. Either:

            * a single :class:`Moment`/:class:`MetricPhaseTarget` — align the *whole* call as one run,
              landing its end on the target; or
            * a per-segment list/tuple (same length as ``beat_length_targets``) of ``None``/targets — each
              non-``None`` entry closes a *run* (all segments since the previous alignment, or the start)
              and bends them collectively so the run lands on that target.

            A group-aligned run of more than one segment must have its ``whens`` on a single axis (all beats
            or all time), with the alignment target on the opposite (free) axis.
        :raises ValueError: if ``whens``/``curve_shapes``/``align_to`` lengths are inconsistent with
            ``beat_length_targets``; if the ``whens`` do not come out strictly increasing in clock-time (the
            error names the offending index); if a multi-segment aligned run mixes beat- and time-axis
            ``whens``; or if an alignment target is unreachable by curvature adjustment. On any failure the
            tempo curve is left unchanged.
        """
        self._apply_targets(self.tempo_history.set_beat_length_target,
                            beat_length_targets, whens, curve_shapes, truncate, align_to)

    @_reschedule_after_tempo_change
    def set_rate_targets(self, rate_targets: Sequence[float],
                         whens: 'Sequence[ResolvableMoment]',
                         curve_shapes: Sequence[float] = None,
                         truncate: bool = True,
                         align_to: 'ResolvableMoment | Sequence[ResolvableMoment | None]' = None) -> None:
        """Smoothly change this clock's rate (beats per second; the reciprocal of beat length, so a higher
        rate is a faster tempo) through a series of targets — the multi-segment form of
        :meth:`set_rate_target`.

        :param rate_targets: the rates (beats per second) to arrive at, one per segment.
        :param whens: when each target is reached. See :meth:`set_beat_length_targets`.
        :param curve_shapes: optional per-segment bends. See :meth:`set_beat_length_targets`.
        :param truncate: whether to discard already-scheduled tempo changes first. See
            :meth:`set_beat_length_targets`.
        :param align_to: optional curvature-solved alignment of segment runs. See
            :meth:`set_beat_length_targets`.
        :raises ValueError: see :meth:`set_beat_length_targets`."""
        self._apply_targets(self.tempo_history.set_rate_target,
                            rate_targets, whens, curve_shapes, truncate, align_to)

    @_reschedule_after_tempo_change
    def set_tempo_targets(self, tempo_targets: Sequence[float],
                          whens: 'Sequence[ResolvableMoment]',
                          curve_shapes: Sequence[float] = None,
                          truncate: bool = True,
                          align_to: 'ResolvableMoment | Sequence[ResolvableMoment | None]' = None) -> None:
        """Smoothly change this clock's tempo (beats per minute) through a series of targets — the
        multi-segment form of :meth:`set_tempo_target`, and usually the most natural of the three
        equivalent plural setters.

        :param tempo_targets: the tempos (beats per minute) to arrive at, one per segment.
        :param whens: when each target is reached. See :meth:`set_beat_length_targets`.
        :param curve_shapes: optional per-segment bends. See :meth:`set_beat_length_targets`.
        :param truncate: whether to discard already-scheduled tempo changes first. See
            :meth:`set_beat_length_targets`.
        :param align_to: optional curvature-solved alignment of segment runs. See
            :meth:`set_beat_length_targets`.
        :raises ValueError: see :meth:`set_beat_length_targets`."""
        self._apply_targets(self.tempo_history.set_tempo_target,
                            tempo_targets, whens, curve_shapes, truncate, align_to)

    @_reschedule_after_tempo_change
    def apply_beat_length_function(self, function: Callable, domain_start: float = 0,
                                   domain_end: float = None, duration_units: str = "beats",
                                   truncate: bool = True, loop: bool = False,
                                   extension_increment: float = 2.0, **kwargs) -> None:
        """Drive this clock's beat_length from a function. See
        :meth:`TempoHistory.apply_function` for the full parameter list (passed via **kwargs)."""
        self.tempo_history.apply_function(
            function, domain_start=domain_start, domain_end=domain_end, units="beatlength",
            duration_units=duration_units, truncate=truncate, loop=loop,
            extension_increment=extension_increment, **kwargs,
        )

    @_reschedule_after_tempo_change
    def apply_rate_function(self, function: Callable, domain_start: float = 0,
                            domain_end: float = None, duration_units: str = "beats",
                            truncate: bool = True, loop: bool = False,
                            extension_increment: float = 2.0, **kwargs) -> None:
        """Drive this clock's rate from a function."""
        self.tempo_history.apply_function(
            function, domain_start=domain_start, domain_end=domain_end, units="rate",
            duration_units=duration_units, truncate=truncate, loop=loop,
            extension_increment=extension_increment, **kwargs,
        )

    @_reschedule_after_tempo_change
    def apply_tempo_function(self, function: Callable, domain_start: float = 0,
                             domain_end: float = None, duration_units: str = "beats",
                             truncate: bool = True, loop: bool = False,
                             extension_increment: float = 2.0, **kwargs) -> None:
        """Drive this clock's tempo (BPM) from a function."""
        self.tempo_history.apply_function(
            function, domain_start=domain_start, domain_end=domain_end, units="tempo",
            duration_units=duration_units, truncate=truncate, loop=loop,
            extension_increment=extension_increment, **kwargs,
        )

    @_reschedule_after_tempo_change
    def apply_tempo_envelope(self, envelope: TempoEnvelope, truncate: bool = True,
                             loop: bool = False) -> None:
        """Append the given :class:`TempoEnvelope` onto this clock's internal tempo envelope, starting from
        the current beat. With `loop=True` the envelope repeats indefinitely until
        :meth:`stop_tempo_loop_or_function`. `truncate` first discards any tempo curve already
        projected past the current beat so the envelope begins cleanly from now."""
        self.tempo_history.append_envelope(envelope, truncate=truncate, loop=loop)

    def stop_tempo_loop_or_function(self) -> None:
        """Stop following any function or looping envelope previously applied to this clock's tempo."""
        self.tempo_history.stop_follow_function_or_envelope_loop()

    def clock_to_scheduler_time(self, beat_or_time, units="beats"):
        """
        Gets the time in the scheduler for a given beat or time in this clock, working recursively up the chain
        of clocks.

        :param beat_or_time: the beat or time of interest in this clock
        :param units: one of ("beats", "time"), determining the units for the first argument
        :return: how much time should pass in the scheduler
        """
        units = DurationUnits(units)
        t = (beat_or_time if units == "time" else self.tempo_history.time_at_beat(beat_or_time)) + self.parent_offset
        for clock in self.inheritance(include_self=False):
            t = clock.tempo_history.time_at_beat(t) + clock.parent_offset
        return t

    def scheduler_to_clock_time(self, scheduler_time, desired_units="beats"):
        """
        Gets beats or time in this clock for a given time in the scheduler.

        :param scheduler_time: the time of interest in the scheduler
        :param desired_units: one of ("beats", "time"), whether we're looking for beats or time in this clock
        :return: how much time should pass in the scheduler
        """
        desired_units = DurationUnits(desired_units)
        t = scheduler_time
        for clock in reversed(self.inheritance(include_self=False)):
            t = clock.tempo_history.beat_at_time(t - clock.parent_offset)
        if desired_units == DurationUnits.TIME:
            return t - self.parent_offset
        else:
            return self.tempo_history.beat_at_time(t - self.parent_offset)

    def bring_up_to_date(self):
        """
        Advance this clock's tempo_history to match where it currently is in scheduler time.

        Called before mutating tempo from a foreign thread: if we don't, the tempo setter's `truncate()`
        cuts at the last *committed* beat (whatever the clock last woke at), causing the new tempo to
        retroactively reshape the segment the clock is currently napping through. From the owning thread
        this is a no-op since committed == current.
        """
        # delta = (live scheduler-derived beat) - (last committed beat in tempo_history)
        delta = self.beat() - self.tempo_history.beat()
        if delta > 0:
            self.tempo_history.advance(delta)

    def extract_absolute_tempo_envelope(self, start_beat: float = 0, step_size: float = 0.1,
                                        tolerance: float = 0.005) -> TempoEnvelope:
        """
        Extract this clock's absolute tempo curve — its tempo as observed in master (scheduler) time,
        with parent rate-changes folded in. Used when building a Score from this clock's perspective.

        For master, this is just the tempo_history as-is. Otherwise we walk the inheritance chain:
        deepcopy each clock's tempo_history, position each at the beat it has when the child is at
        `start_beat`, then step the child forward `step_size` at a time and cascade the resulting
        time-deltas up through each parent. Final delta in master (seconds) / step_size (beats in this clock)
        gives us a sample of absolute beat length, which we use to build up an absolute tempo envelope.
        """
        if self.is_master():
            return self.tempo_history.as_tempo_envelope()

        clocks = self.inheritance()
        tempo_histories = [deepcopy(c.tempo_history) for c in clocks]
        tempo_histories[0].go_to_beat(start_beat)
        initial_rate = tempo_histories[0].rate
        for i in range(1, len(tempo_histories)):
            # parent's beat at this moment = child's parent_offset + child's elapsed time
            tempo_histories[i].go_to_beat(clocks[i - 1].parent_offset + tempo_histories[i - 1].time())
            initial_rate *= tempo_histories[i].rate

        def step_and_get_beat_length(step):
            beat_change = step
            for th in tempo_histories:
                _, beat_change = th.advance(beat_change)
            return beat_change / step

        output_curve = TempoEnvelope(initial_rate, units="rate")
        while any(th.beat() < th.length() for th in tempo_histories):
            # sample twice at half step_size so we can use the midpoint as a curvature guide
            start_level = output_curve.end_level()
            halfway_level = step_and_get_beat_length(step_size / 2)
            end_level = step_and_get_beat_length(step_size / 2)
            if min(start_level, end_level) < halfway_level < max(start_level, end_level):
                output_curve.append_segment(end_level, step_size, tolerance=tolerance,
                                            halfway_level=halfway_level)
            else:
                # midpoint outside [start, end] => turnaround; fall back to two linear segments
                output_curve.append_segment(halfway_level, step_size / 2, tolerance=tolerance)
                output_curve.append_segment(end_level, step_size / 2, tolerance=tolerance)
        return output_curve

    def _reschedule_self_and_descendants(self):
        """
        After a tempo change on self, any queued scheduler-event whose `acting_clock` is self or a
        descendant of self has a stale `t` (computed against the old tempo curve). Recompute the
        scheduler-time of each match from its target Moment, which re-derives it under the new tempo
        while preserving whichever of beat/time the Moment expresses.
        """
        affected = {id(c) for c in self.iterate_descendants(include_self=True)}

        def matches(event):
            meta = event.metadata
            if not isinstance(meta, dict):
                return False
            ac = meta.get("acting_clock")
            return ac is not None and id(ac) in affected and "target_moment" in meta

        def recompute(event):
            meta = event.metadata
            return meta["target_moment"].scheduler_time(meta["acting_clock"])

        self.scheduler.reschedule(matches, recompute)

    ##################################################################################################################
    #                                              Waiting and Forking
    ##################################################################################################################

    def _schedule_at(self, moment: Moment, action: Callable, priority: tuple,
                     description: str = None, extra_metadata: dict = None) -> None:
        """
        Enqueue `action` on the scheduler at the (absolute) `moment` on this clock, tagged so the
        shared machinery handles it: `acting_clock` + `target_moment` let a tempo change reschedule it
        (preserving whichever of beat/time the moment expresses) and let kill() cancel it. This is the
        single scheduling path behind wait(), schedule_action(), and fork().
        """
        meta = {"description": description or f"Action on {self}",
                "acting_clock": self,
                "target_moment": moment}
        if extra_metadata:
            meta.update(extra_metadata)
        self.scheduler.schedule_action(moment.scheduler_time(self), action, priority, meta)

    def wait(self, duration: float | ResolvableMoment, units: str = "beats") -> None:
        """
        Block this clock for `duration` beats (or seconds, if units="time") from now, yielding to the
        scheduler. `duration` may also be any ResolvableMoment (a Moment or MetricPhaseTarget), in
        which case it is resolved directly and `units` is ignored — e.g. wait(Moment.at_beat(8)) or
        wait(MetricPhaseTarget(0, 4)). wait_until() is just a convenience for absolute targets given as
        a bare number; passing an absolute Moment to wait() does the same thing.
        """
        self._wait(to_absolute_moment(duration, self, units_if_number=units, relative_if_number=True))

    def wait_until(self, when: float | ResolvableMoment, units: str = "beats") -> None:
        """
        Block this clock until the beat (or time, if units="time") indicated by `when` — a convenience
        for absolute targets given as a bare number (equivalent to wait(Moment.at_beat(when)), or
        Moment.at_time for units="time"). `when` may also be any ResolvableMoment, resolved directly
        (units ignored). If `when` is in the past, returns essentially immediately.
        """
        self._wait(to_absolute_moment(when, self, units_if_number=units, relative_if_number=False))

    def _wait(self, moment: Moment | None) -> None:
        """
        Underlying implementation for user-facing wait methods: blocks until the (absolute) `moment` on
        this clock. wait() and wait_until() are thin front-ends that resolve their arguments to a Moment
        and call this. `moment=None` means "wait forever" — schedule no wakeup at all and park until the
        clock is killed (used by wait_forever()).

        Four steps:
            (1) reject if this clock isn't ALIVE, or we're not on its own thread;
            (2) unless moment is None, queue the wake-up event (via _schedule_at) for `moment`;
            (3) hand off to the scheduler and block;
            (4) on wake, raise ClockKilledError if killed mid-wait, otherwise advance the
                committed pointer in tempo_history to reflect the post-wait beat/time.

        See in-line comments for details on each step and see the `kill()` docstring for the full picture
        of how clocks terminate.
        """
        # --------------------- STEP 1: Validity checks (state and thread) ---------------------

        if self._state is not ClockState.ALIVE:
            # Mirrors fork()'s rule: wait/fork only allowed on ALIVE clocks. PENDING is rejected
            # because the clock has no thread of its own yet — any wait() call would have to come
            # from some other thread holding the reference, which doesn't make sense.
            raise DeadClockError(
                f"Cannot call wait on a clock that is {self._state.value} (not ALIVE)."
            )

        if current_clock() is not self:
            # wait() blocks the calling thread, so it must be called from this clock's own thread.
            # Otherwise we'd block the wrong thread, schedule a wakeup keyed to this clock's id,
            # and... well it sounds chaotic, and I cannot imagine a legitimate use case.
            raise WrongThreadError(
                f"wait() on {self} must be called from its own thread (use current_clock().wait(...))."
            )

        # ------------------- STEP 2: Schedule the wake-up for `moment` --------------------

        # moment is None signifies an indefinite wait where the only thing that can wake us (below)
        # is kill() setting our _wait_event. Otherwise queue the wake-up that fires when we reach `moment`.
        if moment is not None:
            self._schedule_at(moment, self._wake_and_advance_to_next_wait_call, self._priority,
                              description=f"{self} wakeup action")

        # ---------------------------- STEP 3: Hand off to the scheduler -----------------------------

        # Clear _wait_event, then release the scheduler. Order matters: once we notify,
        # the scheduler is free to fire our wake-up (which sets _wait_event), and a clear
        # after that would wipe the signal.
        self._wait_event.clear()
        with self._scheduler_park_condition:
            self._scheduler_park_condition.notify_all()

        # THIS IS WHERE OTHER THREADS TAKE OVER
        # when it's time to wake up, the scheduler runs the scheduled _wake_and_advance_to_next_wait_call
        # which releases us from this wait, and then enters its own wait condition, waiting for the next
        # wait call on this clock to notify (right above this). (On the master's very first wait, the
        # scheduler is parked on this same condition by the initial _wake_and_advance scheduled from
        # __init__; on a sub-clock's first wait, by _start_new_clock after launching the clock's thread.)
        self._wait_event.wait()

        # ----------------------------------- STEP 4: Clean Up --------------------------------------

        # ~~~~~ Step 4a: Handle clock woken by kill() ~~~~~
        # Reaching this branch means kill() set _wait_event to wake us; kill() also released the
        # scheduler from _scheduler_park_condition in the same pass, so we don't need to do that here.
        if self._state is ClockState.DEAD:
            raise ClockKilledError()

        # ~~~~~ Step 4b: Advance tempo_history's committed pointer to the beat we woke at ~~~~~
        # bring_up_to_date() advances self.tempo_history's committed pointer to self.beat(), the live
        # scheduler-derived position. Note that since were polling self.beat() now, this is robust even
        # in the case of a time-based wakeup where the tempo has changed since it was scheduled.
        # The internal >0 guard also keeps a past/now target from rewinding.
        self.bring_up_to_date()

    def _wake_and_advance_to_next_wait_call(self):
        """
        This function is run on the scheduler thread: it wakes up this clock, and then blocks until the clock
        has hit another wait call and scheduled its next wakeup.
        """
        with self._scheduler_park_condition:
            self._wait_event.set()
            self._scheduler_park_condition.wait()

    def _prewarm_pool(self, n: int) -> None:
        """
        Eagerly spin up `n` persistent pool workers so the first `n` forks skip the ~60us
        thread-creation latency (a warm fork is ~3x cheaper). Each warm-up task parks on a Barrier,
        which forces the executor to create fresh workers until n exist. Then the Barrier releases
        and the workers return to the pool idle. Uses only public API (submit + Barrier), and bypasses
        _pool_semaphore, since creation and release from the Barrier takes very little time.

        NB: it is unclear this is actually worth doing. The lazy pool already amortizes creation cost
        to ~zero after the first few forks, and the per-fork steady-state win comes from worker *reuse*,
        not from prewarming; prewarming only shaves the one-time ramp, at the cost of giving every clock
        a small fixed thread/latency footprint at construction. Default is small (10) and `prewarm_pool=0`
        disables it. Revisit if benchmarking shows a perceptible hitch at start of playback.

        One point in its favor: thread creation isn't flat — the *first* several threads take longer to
        create, so prewarming front-loads the most expensive creations. The total is still sub-millisecond
        and one-time, but it's why the default is left non-zero rather than 0.
        """
        if n <= 0:
            return
        barrier = threading.Barrier(n + 1)
        for _ in range(n):
            self._pool.submit(barrier.wait)
        barrier.wait()  # returns once all n workers have spun up and reached the barrier

    def _run_in_pool(self, target: Callable, args: Sequence | None, kwargs: dict | None) -> None:
        """
        Run `target` on the family's shared thread pool (owned by the master). Backs fork().
        If the pool is fully occupied, fall back to a plain daemon Thread and warn, rather than
        blocking the caller.
        """
        master = self.master
        kwargs = {} if kwargs is None else kwargs
        args = () if args is None else args
        semaphore = master._pool_semaphore
        if semaphore.acquire(blocking=False):
            # The done-callback fires exactly once when the future finishes (completed, errored, or
            # canceled), so the semaphore is released exactly once (no double-release on a BoundedSemaphore).
            def _on_done(future):
                semaphore.release()
                # A canceled future (only happens via shutdown(cancel_futures=True) in kill()) has no
                # exception to report; otherwise surface any error the task raised.
                if not future.cancelled() and future.exception() is not None:
                    _threadpool_error_callback(future.exception())

            master._pool.submit(target, *args, **kwargs).add_done_callback(_on_done)
        else:
            logging.warning("Ran out of threads in the master clock's thread pool; small thread-creation "
                            "delays may result. You can increase the number of threads via the master "
                            "clock's `pool_size` argument.")
            threading.Thread(target=target, args=args, kwargs=kwargs, daemon=True).start()

    def fork(self, forked_function: Callable, args: Sequence = (), kwargs: dict = None, name: str = None,
             initial_rate: float = None, initial_tempo: float = None, initial_beat_length: float = None,
             when: ResolvableMoment | None = None, done_callback: Callable[[], None] = None):
        """
        Spawns a child clock running `forked_function` as a coordinated parallel timeline.

        The child runs on its own thread but stays synchronized under this clock's master scheduler —
        "parallel" here means parallel musical time (like a separate voice or layer), not simultaneous
        CPU execution; the scheduler runs one clock's code at a time.

        :param forked_function: the function to be run on the new child clock
        :param args: positional arguments to be passed to the forked function. (Unlike legacy clockblocks,
            clockblocks does *not* inject the child clock as an extra first argument when the signature is one short
            — call :func:`current_clock` from inside the forked function if you need a reference to it.)
        :param kwargs: keyword arguments to be passed to the forked function
        :param name: name to be given to the spawned child clock
        :param initial_rate: starting rate of this clock (if set, don't set initial tempo or beat length)
        :param initial_tempo: starting tempo of this clock (if set, don't set initial rate or beat length)
        :param initial_beat_length: starting beat length of this clock (if set, don't set initial tempo or rate)
        :param when: when the forked function should begin, as a :class:`~clockblocks.moment.ResolvableMoment`.
            None (default) starts it immediately. Otherwise pass an explicit Moment — :meth:`Moment.at_beat`
            (or :meth:`Moment.at_time`) for an absolute point, or :meth:`Moment.after_beats`
            (or :meth:`Moment.after_time`) for an offset from now. Unlike wait and wait_until, a bare
            number is rejected here, since it's not clear whether it would be relative or absolute. Also possible
            is a :class:`~clockblocks.metric_phase.MetricPhaseTarget` which starts it at the next matching point
            in a cycle.
        :param done_callback: a callback function to be invoked when the clock has terminated
        :return: the spawned child clock
        """
        # The state check, child registration, and fork-event scheduling all run under _tree_lock so
        # they're atomic against kill(). Either kill wins (we then see DEAD and raise) or we win (kill
        # then sees the child in the tree and cancels its fork event) — never a half state where the
        # child is scheduled but invisible to a concurrent kill, which would orphan it.
        with self._tree_lock:

            # --------------------- STEP 1: Validity check (state only) ---------------------

            # Unlike wait(), fork() is *not* thread-restricted: it doesn't block, it just queues a
            # scheduler event to start the new child clock, and then returns. The forked _fork_wrapper
            # runs on its own new thread and sets that thread's __clock__ to the new child, so
            # current_clock() works correctly inside it regardless of who called fork().
            if self._state is not ClockState.ALIVE:
                raise DeadClockError(
                    f"Cannot call fork from a clock that is {self._state.value} (not ALIVE)."
                )

            # ------------- STEP 2: Create the child clock and resolve when it should start -------------

            name = (forked_function.__name__ if hasattr(forked_function, '__name__') else "UNNAMED") \
                if name is None else name

            child = Clock(name, parent=self, initial_rate=initial_rate, initial_tempo=initial_tempo,
                          initial_beat_length=initial_beat_length)
            self._children.append(child)

            # Resolve `when` to an absolute moment on this (the parent) clock; the scheduled fork event
            # fires at that moment (and gets rescheduled with it if the tempo changes — see _schedule_at).
            # allow_number=False: a bare number's relative/absolute meaning is ambiguous here, so require
            # an explicit Moment (None still means "now").
            start_moment = to_absolute_moment(when, self, allow_number=False)

            # ------------------- STEP 3: Define the child's lifecycle wrapper (_fork_wrapper) -------------------

            def _fork_wrapper(*args, **kwds):
                # ~~~~~ Wrapper step 1: Thread/clock setup ~~~~~
                # Bind __clock__ so current_clock() resolves to the child inside the user function,
                # finalize parent_offset / _start_time_in_scheduler, and flip ALIVE. Both are read now,
                # at the actual fire instant, so they reflect where the child truly starts — correct no
                # matter how the fork event was rescheduled in the interim (e.g. by a tempo change).
                threading.current_thread().__clock__ = child
                child.parent_offset = self.beat()
                child._start_time_in_scheduler = self.scheduler.time()
                child._state = ClockState.ALIVE

                try:
                    # ~~~~~ Wrapper step 2: Run the user function ~~~~~
                    # ClockKilledError fires from inside wait() when kill() wakes us mid-wait.
                    # DeadClockError is also possible if a thread from outside the clock system kills the clock
                    # while it's awake and in the middle of running user code: when the user code finishes what
                    # it was doing and reaches its next wait call, it's calling wait on a dead clock, which
                    # raises DeadClockError.
                    try:
                        forked_function(*args, **kwds)
                    except (ClockKilledError, DeadClockError):
                        pass

                    # ~~~~~ Wrapper step 3: Cleanup ~~~~~
                    # Detach from the forking parent (self) and mark DEAD, under _tree_lock so it's
                    # consistent with kill()'s detach and with any concurrent fork/kill. This is for natural
                    # exits and is redundant for killed clocks.
                    with self._tree_lock:
                        # _detach_child does the list removal and, if this was the last child of a parent
                        # blocked in wait_for_children_to_finish(), releases it.
                        self._detach_child(child)
                        child._state = ClockState.DEAD

                    # ~~~~~ Wrapper step 4: Release the scheduler ~~~~~
                    # After the user function returns, the scheduler is parked on _scheduler_park_condition
                    # (either from the most recent _wake_and_advance_to_next_wait_call, or — if the user function
                    # never called wait — from _start_new_clock). Notify it so it can proceed to the next event.
                    # Again, this is redundant for killed clocks, as the _scheduler_park_condition will already
                    # have been notified from within kill()
                    with child._scheduler_park_condition:
                        child._scheduler_park_condition.notify_all()

                    # ~~~~~ Wrapper Step 5: done_callback if requested by user ~~~~~
                    if done_callback is not None:
                        done_callback()
                finally:
                    # ~~~~~ Wrapper step 6: Untag the (pooled, reused) worker thread ~~~~~
                    # Drop this thread's reference to the now-dead child so an idle pool worker doesn't pin
                    # it (and its tempo_history etc.) alive until its next task. Done last, in a finally, so
                    # done_callback still sees current_clock() == child and so it runs even on an error path.
                    # The next task on this worker re-tags __clock__ before reading it, so None is safe here.
                    threading.current_thread().__clock__ = None

            # ---------- STEP 4: Define the launcher, schedule it, and return the child ----------

            def _start_new_clock():
                # NB: This is run from the scheduler (scheduled below), so the code should be understood from that
                # perspective. It is the *scheduler* (and not the parent clock's thread!) that launches the forked
                # function at the scheduled time, and it's the *scheduler* that is parking on the new clock's
                # _scheduler_park_condition, waiting to be freed by the first wait call in the new clock
                # (or the cleanup in _fork_wrapper, if that function never calls wait).

                if child._state is ClockState.DEAD:
                    # If kill() was called during the start_delay window, just skip starting the thread.
                    # kill() will have removed this event in most cases, but guard against the race.
                    return
                with child._scheduler_park_condition:
                    self._run_in_pool(_fork_wrapper, args, kwargs)
                    child._scheduler_park_condition.wait()

            self._schedule_at(start_moment, _start_new_clock, child._priority,
                              description=f"Forking of {child}",
                              extra_metadata={"forked_child": child})
            return child

    def schedule_action(self, action: Callable, when: ResolvableMoment,
                        args: Sequence = (), kwargs: dict = None) -> None:
        """
        The lightweight alternative to fork() for immediately returning functions. Schedules `action` to
        run once at `when`, as a leaf event on the scheduler thread — without spawning a child clock.
        For functions that don't wait, this is much more performant: there is no thread creation, no
        scheduler/clock handoff; just a callable fired at the right musical time.

        For work that needs to wait, fork() a child clock instead. The function scheduled here will not
        have an active clock to wait on, and will hold up the entire scheduler if it sleeps.

        :param action: the callable to run (exceptions are caught and logged by the scheduler)
        :param when: when to run it, as a :class:`~clockblocks.moment.ResolvableMoment` — same convention as
            fork(): an explicit Moment (Moment.at_beat/at_time for an absolute point, Moment.after_beats/
            after_time for an offset from now) or a MetricPhaseTarget. Unlike wait and wait_until, a bare
            number is rejected here, since it's not clear whether it would be relative or absolute.
        :param args: positional arguments to pass to action
        :param kwargs: keyword arguments to pass to action
        """
        kwargs = {} if kwargs is None else kwargs

        # Atomic against kill(), exactly like fork(): under _tree_lock, either we observe DEAD and
        # refuse, or we push the event and a concurrent kill() then finds and cancels it (kill()'s
        # matcher keys on acting_clock, which is self).
        with self._tree_lock:
            if self._state is not ClockState.ALIVE:
                raise DeadClockError(
                    f"Cannot schedule_action on a clock that is {self._state.value} (not ALIVE)."
                )
            moment = to_absolute_moment(when, self, allow_number=False)
            self._schedule_at(moment, lambda: action(*args, **kwargs), self._priority,
                              description=f"Scheduled action on {self}")

    def wait_forever(self) -> None:
        """
        Block this clock's own thread indefinitely, yielding to the scheduler so child clocks keep
        running. Typically called once a clock has forked its work and has nothing left to do itself
        (e.g. the master clock keeping the main thread alive).

        This only ever unblocks via :meth:`kill`, at which point it raises :class:`ClockKilledError`.
        For a forked clock, this is caught by the fork wrapper and it cleanly unwinds.
        For a master clock it would need to be caught. A built-in way to do this is by using the clock
        as a context manager, which does exception handling and teardown for you. (:meth:`run_as_server`
        also absorbs the exception automatically within the spawned thread).

        That said, typically this is used at end of script and killed via ctrl-c/process exit, so no
        explicit catching of this exception is necessary.
        """
        # Implemented as a single _wait(None): no wakeup is scheduled,
        # so the clock simply parks until kill() wakes it.
        self._wait(None)

    def wait_for_children_to_finish(self) -> None:
        """
        Block this clock's own thread until all of its child clocks have finished, yielding to the
        scheduler so they can run, then return as soon as the last child ends.

        If this clock is itself killed while waiting, raises :class:`ClockKilledError` rather than
        returning — so a normal return always means the children genuinely finished.
        """
        # Park on an indefinite _wait(None) (no scheduled wake-up); _detach_child watches
        # _waiting_for_children and schedules the wakeup that releases us when the last child detaches.
        with self._tree_lock:
            # _tree_lock guards against a non-clock thread killing the last child in between checking for
            # children and setting self._waiting_for_children = True. If this happened, the _detach_child
            # call from kill() would see _waiting_for_children still False and so schedule no releasing
            # wakeup; we'd then set the flag True and park on _wait(None) with no child left to release us
            if not self._children:
                return
            self._waiting_for_children = True
        try:
            self._wait(None)
        finally:
            self._waiting_for_children = False

    def run_as_server(self) -> 'Clock':
        """
        Run this (master) clock on a background daemon thread so the calling thread stays free — the
        approach for driving clockblocks from an interactive REPL: ``c = Clock().run_as_server()``.
        The background thread becomes the clock's owning thread and waits forever; the calling thread
        relinquishes ownership (its `current_clock()` becomes None), so further work must be forked on
        the returned clock object directly (``c.fork(...)``), not via the module-level helpers. Returns
        self. Only valid on the master clock — raises NotMasterClockError on a child.
        """
        if not self.is_master():
            raise NotMasterClockError(
                f"run_as_server() is only valid on the master clock, not {self}.")

        def run_server():
            threading.current_thread().__clock__ = self
            # run_as_server is a managed entry point, so it absorbs the kill: wait_forever() now raises
            # ClockKilledError when the clock is killed, and we swallow it here so the server thread
            # exits quietly instead of dumping a traceback through the thread excepthook.
            try:
                self.wait_forever()
            except (ClockKilledError, DeadClockError):
                pass

        threading.Thread(target=run_server, daemon=True).start()
        # The calling thread no longer owns this clock.
        threading.current_thread().__clock__ = None
        return self

    @property
    def alive(self) -> bool:
        """True if this clock is currently running (PENDING and DEAD both return False)."""
        return self._state is ClockState.ALIVE

    ##################################################################################################################
    #                                                 Fast-forwarding
    ##################################################################################################################

    # Fast-forwarding is a scheduler-wide operation: the shared scheduler fires queued events with no
    # wall-clock waiting, so every wait() in the family returns instantaneously and musical time races
    # ahead. The goal is stored on the scheduler as a *scheduler-time*; the methods below convert this
    # clock's requested time/beat into scheduler-time via clock_to_scheduler_time.

    def fast_forward(self, on_or_off: bool = True) -> None:
        """
        Turn indefinite fast-forwarding on or off. While on, all waiting is instantaneous.
        (Only valid on the master clock.)

        :param on_or_off: True to start fast-forwarding, False to stop.
        """
        if not self.is_master():
            raise NotMasterClockError("Only the master clock can be fast-forwarded.")
        self.scheduler.set_fast_forward_goal(float("inf") if on_or_off else None)

    def fast_forward_to_time(self, t: float) -> None:
        """
        Fast-forward, skipping instantaneously up to time `t` (in seconds) on this clock, then resume
        real-time playback. (Only valid on the master clock.)

        :param t: time to fast-forward to
        """
        if not self.is_master():
            raise NotMasterClockError("Only the master clock can be fast-forwarded.")
        if t < self.time():
            raise ValueError("Cannot fast-forward to a time in the past.")
        if math.isinf(t):
            self.scheduler.set_fast_forward_goal(float("inf"))
            return
        self.scheduler.set_fast_forward_goal(self.clock_to_scheduler_time(t, units="time"))

    def fast_forward_in_time(self, t: float) -> None:
        """
        Fast-forward, skipping ahead instantaneously by `t` seconds. (Only valid on the master clock.)

        :param t: number of seconds to fast-forward by
        """
        self.fast_forward_to_time(self.time() + t)

    def fast_forward_to_beat(self, b: float) -> None:
        """
        Fast-forward, skipping instantaneously up to beat `b` on this clock. (Only valid on the master clock.)

        :param b: beat to fast-forward to
        """
        if not self.is_master():
            raise NotMasterClockError("Only the master clock can be fast-forwarded.")
        if b < self.beat():
            raise ValueError("Cannot fast-forward to a beat in the past.")
        if math.isinf(b):
            self.scheduler.set_fast_forward_goal(float("inf"))
            return
        self.scheduler.set_fast_forward_goal(self.clock_to_scheduler_time(b, units="beats"))

    def fast_forward_in_beats(self, b: float) -> None:
        """
        Fast-forward, skipping ahead instantaneously by `b` beats. (Only valid on the master clock.)

        :param b: number of beats to fast-forward by
        """
        self.fast_forward_to_beat(self.beat() + b)

    def is_fast_forwarding(self) -> bool:
        """
        Whether the clock is currently fast-forwarding. Since fast-forwarding is a scheduler-wide state,
        this is true for every clock in the family whenever it's true for any of them.
        """
        return self.scheduler.is_fast_forwarding()

    @property
    def timing_policy(self) -> float:
        """
        How the family trades off relative vs. absolute timing, as a float from 0 to 1.

        At 1.0 (relative) each wait is kept as faithful as possible to its requested duration, timed from
        when the previous event fired (callback runtime does not count against the wait). This can still let
        the clock fall behind real time: if a wait overruns (e.g. through a callback that runs longer than its
        own wait or the OS waking late) that lateness is never made up. At 0.0 (absolute) the clock instead stays
        faithful to the time elapsed since it began — a wait that ran long is followed by shorter waits to
        catch up, at the cost of some relative-timing accuracy. A value in between clamps the catch-up: the
        clock attempts to stay on the absolute schedule but can only compress a wait to `wait_dur * timing_policy`
        (or stretch it to `wait_dur / timing_policy` if trying to compensate for being ahead).
        (Forwards to the scheduler, settable only on the master clock)
        """
        return self.scheduler.timing_policy

    @timing_policy.setter
    def timing_policy(self, value: float) -> None:
        if not self.is_master():
            raise NotMasterClockError(
                "timing_policy applies to the whole clock family; set it on the master clock."
            )
        if not 0.0 <= value <= 1.0:
            raise ValueError("timing_policy must be between 0 (absolute) and 1 (relative).")
        self.scheduler.timing_policy = value

    def use_absolute_timing_policy(self) -> None:
        """Shorthand for ``timing_policy = 0.0`` (always catch up to absolute schedule)."""
        self.timing_policy = 0.0

    def use_relative_timing_policy(self) -> None:
        """Shorthand for ``timing_policy = 1.0`` (always wait the full requested delay)."""
        self.timing_policy = 1.0

    def use_mixed_timing_policy(self, absolute_relative_mix: float) -> None:
        """Shorthand for ``timing_policy = absolute_relative_mix`` (a blend between 0=absolute and 1=relative)."""
        self.timing_policy = absolute_relative_mix

    @property
    def precise_timing(self) -> bool:
        """
        Whether to use a busy wait in the final moments leading up to a scheduled event. The busy wait fully
        occupies a CPU core and lasts at most :attr:`spin_guard_duration` seconds per event. (Forwards to
        the scheduler, settable only on the master clock)
        """
        return self.scheduler.precise_timing

    @precise_timing.setter
    def precise_timing(self, value: bool) -> None:
        if not self.is_master():
            raise NotMasterClockError(
                "precise_timing applies to the whole clock family; set it on the master clock.")
        self.scheduler.precise_timing = value

    @property
    def spin_guard_duration(self) -> float:
        """
        Width (seconds) of the busy-wait guard band used when :attr:`precise_timing` is on. (Forwards to
        the scheduler, settable only on the master clock)
        """
        return self.scheduler.spin_guard_duration

    @spin_guard_duration.setter
    def spin_guard_duration(self, value: float) -> None:
        if not self.is_master():
            raise NotMasterClockError(
                "spin_guard_duration applies to the whole clock family; set it on the master clock.")
        if value < 0:
            raise ValueError("spin_guard_duration must be non-negative.")
        self.scheduler.spin_guard_duration = value

    def kill(self) -> None:
        """
        End this clock (and the corresponding forked function if not master) and cascade to all descendant clocks.

        Pending scheduled work for this clock and its descendants is cancelled and the clocks are
        marked dead. A clock currently blocked in :meth:`wait` raises :class:`ClockKilledError`; any
        later :meth:`wait` or :meth:`fork` on a dead clock raises :class:`DeadClockError`. Killing the
        master also tears down the family's scheduler.

        Safe to call from any thread, and killing an already-dead clock is a no-op.

        (See the "Clock termination / lifecycle paths" note near the top of this module for how the
        four termination paths work internally.)
        """
        if self._state is ClockState.DEAD:
            return

        # Everything below runs under _tree_lock so it's atomic against a concurrent fork(). This is important
        # because we look at the state of the tree and then take action on it; if a fork could arrive between
        # walking the tree and implementing the kill (flag as DEAD, cancel events, detach from parent), we might
        # miss a just-forked child.
        # With the tree lock, either fork arrives fully before (and therefore we see the new child and kill it) or
        # fork arrives fully after (and therefore raises a DeadClockError).
        with self._tree_lock:
            if self._state is ClockState.DEAD:
                # Lost the race to another kill() while acquiring the lock; it did our work.
                return

            # ---------------------- STEP 1: Collect victims (this and descendants) --------------------

            # Collect everyone we're killing up front so the predicate sees a consistent set.
            # lol that Claude called these victims.
            victims = list(self.iterate_descendants(include_self=True))
            victim_ids = {id(c) for c in victims}

            # ----- STEP 2: Flag victims as killed and remove all victim events from the scheduler -----

            def matches(event):
                # Predicate function for picking out the events on the scheduler that should no longer happen.
                # Note that the "forked_child" check covers cases where a victim has called fork with schedule_at
                # sometimes in the future. That clock doesn't exist yet, and this prevents the event that creates it
                meta = event.metadata
                if not isinstance(meta, dict):
                    return False
                ac = meta.get("acting_clock")
                fc = meta.get("forked_child")
                return (ac is not None and id(ac) in victim_ids) or \
                       (fc is not None and id(fc) in victim_ids)

            for c in victims:
                c._state = ClockState.DEAD
            self.scheduler.remove_events(matches)

            # ---- STEP 3: For each victim, wake it (or release scheduler), and detach it from its parent ----

            for c in victims:
                # if the victim is parked mid-wait, wake it: it will observe DEAD, raise ClockKilledError
                c._wait_event.set()
                # If a victim is mid-execution (running user code between waits), the scheduler
                # is parked on its _scheduler_park_condition right now. Free it immediately.
                # For a sub clock it would eventually be released by _fork_wrapper cleanup, but that will
                # only happen when the user code hits the next wait and throws a DeadClockError
                # (which could be arbitrarily long & hold up scheduled events). For the master clock
                # this is critical, because it's not wrapped in _fork_wrapper and would never otherwise release
                # the scheduler.
                with c._scheduler_park_condition:
                    c._scheduler_park_condition.notify_all()
                # Detach from parent so the parent's _children list is accurate from the moment kill()
                # returns. _fork_wrapper would normally do this on cleanup, but that can be delayed (user code
                # between waits has to reach its next wait first) or skipped entirely (a PENDING victim
                # killed during start_delay never has _fork_wrapper run at all). If the parent isn't itself
                # a victim and was blocked in wait_for_children_to_finish(), _detach_child releases it
                # (a parent that *is* a victim is already DEAD here, so it gets no spurious wake).
                if c.parent is not None:
                    c.parent._detach_child(c)

        # ------------------------------ STEP 4: Kill scheduler and pool -------------------------------
        # Clean up the scheduler and thread pool when killing the master.
        # (No need to hold tree lock here anymore; it's irrelevant)
        if self.is_master():
            self.scheduler.kill()
            # All victims have been flagged DEAD, woken, and detached above, so any pooled fork workers
            # have already unwound (or are about to). shutdown(wait=False) lets the executor's workers
            # exit once their current task returns without blocking us here; cancel_futures drops anything
            # still queued (there shouldn't be any — the semaphore caps submissions at the worker count).
            self._pool.shutdown(wait=False, cancel_futures=True)

    def __enter__(self) -> 'Clock':
        """
        Use this clock — typically a master clock or a scamp :class:`~scamp.session.Session` — as a context
        manager: ``with Session() as s: ...``. Returns self; see :meth:`__exit__` for what leaving does.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        On leaving the ``with`` block, :meth:`kill` this clock — which, for a master, also tears down
        its scheduler thread (the two are 1:1). This guarantees cleanup even when the clock runs off the
        main thread, where simply falling off the end would otherwise leak the parked scheduler daemon.

        A :class:`ClockKilledError` / :class:`DeadClockError` propagating out of the block (e.g. an
        external ``kill()`` interrupted a wait) is suppressed — being killed is a clean way for a managed
        clock to end. Any other exception propagates normally (after the clock is killed).
        """
        self.kill()
        return exc_type is not None and issubclass(exc_type, (ClockKilledError, DeadClockError))

    ##################################################################################################################
    #                                            Removed Legacy APIs
    ##################################################################################################################
    # APIs from clockblocks 0.x that no longer exist in clockblocks 1.x, kept as raise-on-access stubs so
    # users porting old code get an actionable message instead of a bare AttributeError.
    # ------------------------------------------------------------------

    @staticmethod
    def _removed_attribute(name: str, replacement: str, reason: str):
        raise AttributeError(
            f"{name} was removed in clockblocks 1.0: {reason}. Use {replacement} instead."
        )

    @property
    def synchronization_policy(self):
        Clock._removed_attribute(
            "Clock.synchronization_policy", "(no replacement needed)",
            "Clock.beat()/time() now read live scheduler-derived positions from any thread, so there "
            "is nothing to synchronize between sibling clocks"
        )

    @synchronization_policy.setter
    def synchronization_policy(self, value):
        Clock._removed_attribute(
            "Clock.synchronization_policy", "(no replacement needed)",
            "Clock.beat()/time() now read live scheduler-derived positions from any thread, so there "
            "is nothing to synchronize between sibling clocks"
        )

    def rouse_and_hold(self, *args, **kwargs) -> None:
        Clock._removed_attribute(
            "Clock.rouse_and_hold()", "`with clock.while_scheduler_quiescent(): ...`",
            "the rouse half is obsolete (lazy beat()/time() are live from any thread), and the hold "
            "half is now an exception-safe `with` block that pairs acquire/release automatically"
        )

    def release_from_suspension(self, *args, **kwargs) -> None:
        Clock._removed_attribute(
            "Clock.release_from_suspension()", "`with clock.while_scheduler_quiescent(): ...`",
            "the rouse/hold pair is now an exception-safe `with` block"
        )

    def __repr__(self):
        child_list = "" if len(self._children) == 0 else ", ".join(str(child) for child in self._children)
        return ("Clock('{}')".format(self.name) if self.name is not None else "UNNAMED") + "[" + child_list + "]"