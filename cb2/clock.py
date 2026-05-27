import functools
import threading
from enum import Enum
from itertools import count
from typing import Callable, Sequence, Iterator
from cb2.tempo_envelope import TempoHistory
from cb2.scheduler import get_scheduler, Scheduler
from cb2.moment import Moment, ResolvableMoment, to_absolute_moment
from cb2.enums import DurationUnits
from cb2.utilities import _PrintColors, current_clock, _spawn_unsynchronized
import textwrap


class ClockblocksError(Exception):
    """Base class for clockblocks errors."""
    pass


class ClockKilledError(ClockblocksError):
    """Raised inside a forked clock's own thread when its wait is woken by kill(),
    so the fork wrapper can unwind the user function cleanly."""
    pass


class DeadClockError(ClockblocksError):
    """Raised when something tries to wait or fork on a clock that's no longer ALIVE
    (either killed already, or — for fork — still PENDING in its start_delay)."""
    pass


class WrongThreadError(ClockblocksError):
    """Raised when wait() is called from a thread that doesn't own the clock.
    Each clock has exactly one owning thread (the one running its forked function,
    or the main thread for the master); calling wait() from any other thread would
    block the wrong thread and corrupt the clock's bookkeeping."""
    pass


class NoActiveClockError(ClockblocksError):
    """Raised when a clock operation (the module-level wait/fork/etc.) is attempted from a thread
    that has no clock active on it. Establish one with fork() / run_as_server(), or call the method
    on a Clock object directly. (Threads spawned by fork_unsynchronized are exempt for the
    sleep-based waits — see fork_unsynchronized.)"""
    pass


class NotMasterClockError(ClockblocksError):
    """Raised by operations that are only valid on the master (top-level) clock — e.g.
    run_as_server() — when called on a child clock."""
    pass


class ClockState(Enum):
    PENDING = "pending"  # forked, awaiting start_delay; thread not yet running user code
    ALIVE = "alive"      # forked_function is running (or about to run)
    DEAD = "dead"        # killed, or forked_function returned


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


class Clock:

    def __init__(self, name: str = None, parent: 'Clock' = None, initial_rate: float = None,
                 initial_tempo: float = None, initial_beat_length: float = None, scheduler: Scheduler = None):
        self.name = name
        self.parent = parent
        self._children = []
        # Counter handed out to *this* clock's future children; their clock_id suffix
        # comes from this counter so siblings get distinct, monotonically-increasing ids.
        self._child_counter = count()
        self.clock_id = (0,) if parent is None else parent.clock_id + (next(parent._child_counter),)

        # tempo envelope, in seconds since I was created
        self.tempo_history = TempoHistory(
            Clock._rate_tempo_or_beat_length_to_rate(initial_rate, initial_tempo, initial_beat_length),
            units="rate"
        )

        # get a default (shared) scheduler unless one is specifically provided
        self.scheduler = self.parent.scheduler if self.parent is not None else \
            get_scheduler() if scheduler is None else scheduler

        self._scheduler_park_condition = threading.Condition()
        self._wait_event = threading.Event()
        # Set while this clock is parked in wait_for_children_to_finish(). It scheduled no wakeup of its
        # own (an indefinite _wait(None)), so _detach_child watches this flag and, when the last child
        # detaches, schedules the wakeup that releases it. See _detach_child / wait_for_children_to_finish.
        self._waiting_for_children = False

        if self.is_master():
            # The whole family shares one structural lock, owned by the master. fork / kill / external
            # tempo changes acquire it (via _tree_lock) so the clock tree (_children, _state) and any
            # enumeration of it stays invariant for the duration of a structural operation.
            self._clock_tree_lock = threading.RLock()
            # Master starts ALIVE (no fork / start_delay)
            self._state = ClockState.ALIVE
            # the first thing we do is stop and put things in the scheduler's hands
            # tell the scheduler to wake up right away and get this clock going, then wait for the scheduler to do it
            self.scheduler.schedule_action(self.scheduler.time(), self._wake_and_advance_to_next_wait_call,
                                           (0, ), {"description": f"Initial wake for {self}", "acting_clock": self})
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
                                           self.clock_id,
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

    def time(self) -> float:
        """
        How much time has passed since this clock was created.
        Either in seconds, if this is the master clock, or in beats in the parent clock, if this clock was the result
        of a call to fork.

        Computed lazily from the current scheduler position, so it stays correct when called from any thread —
        including while the owning thread is mid-wait. Does not mutate `tempo_history` (the committed pointer).
        """
        return self.scheduler_to_clock_time(self.scheduler.time(), desired_units="time")

    def beat(self) -> float:
        """
        How many beats have passed since this clock was created. See `time()` for thread/laziness semantics.
        """
        return self.scheduler_to_clock_time(self.scheduler.time(), desired_units="beats")

    def wall_time_in_scheduler(self) -> float:
        """
        How long has this clock been alive in the scheduler. Should return a result very close to `Clock.time`
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
            self._schedule_at(moment, self._wake_and_advance_to_next_wait_call, self.clock_id,
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

    def fork(self, forked_function: Callable, args: Sequence = (), kwargs: dict = None, name: str = None,
             initial_rate: float = None, initial_tempo: float = None, initial_beat_length: float = None,
             when: ResolvableMoment | None = None, done_callback: Callable[[], None] = None):
        """
        Spawns a child clock running `forked_function` as a coordinated parallel timeline.

        The child runs on its own thread but stays synchronized under this clock's master scheduler —
        "parallel" here means parallel musical time (like a separate voice or layer), not simultaneous
        CPU execution; the scheduler runs one clock's code at a time.

        :param forked_function: the function to be run on the new child clock
        :param args: arguments to be passed to the forked function. One subtlety to note here: if the number of
            arguments passed is one fewer than the number taken by the function, the clock on which the function is
            forked will be passed as the first argument, followed by the arguments given. For instance, if we define
            "forked_function(clock, a, b)", and then call "parent.fork(forked_function, (13, 6))", 13 will be passed
            to "a" and 6 to "b", while the clock on which forked_function is running will be passed to "clock". On the
            other hand, if the signature of the function were "forked_function(a, b)", 13 would be simply be passed to
            "a" and 6 to "b".
        :param kwargs: keyword arguments to be passed to the forked function
        :param name: name to be given to the spawned child clock
        :param initial_rate: starting rate of this clock (if set, don't set initial tempo or beat length)
        :param initial_tempo: starting tempo of this clock (if set, don't set initial rate or beat length)
        :param initial_beat_length: starting beat length of this clock (if set, don't set initial tempo or rate)
        :param when: when the forked function should begin, as a :class:`~cb2.moment.ResolvableMoment`.
            None (default) starts it immediately. Otherwise pass an explicit Moment — :meth:`Moment.at_beat`
            (or :meth:`Moment.at_time`) for an absolute point, or :meth:`Moment.after_beats`
            (or :meth:`Moment.after_time`) for an offset from now. Unlike wait and wait_until, a bare
            number is rejected here, since it's not clear whether it would be relative or absolute. Also possible
            is a :class:`~cb2.metric_phase.MetricPhaseTarget` which starts it at the next matching point in a cycle.
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

                # ~~~~~ Wrapper step 2: Run the user function ~~~~~
                # ClockKilledError fires from inside wait() when kill() wakes us mid-wait.
                # DeadClockError is also possible if a thread from outside the clock system kills the clock while
                # it's awake and in the middle of running user code: when the user code finishes what it was doing
                # and reaches its next wait call, it's calling wait on a dead clock, which raises DeadClockError.
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
                    threading.Thread(target=_fork_wrapper, args=args, kwargs=kwargs, daemon=True).start()
                    child._scheduler_park_condition.wait()

            self._schedule_at(start_moment, _start_new_clock, child.clock_id,
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
        :param when: when to run it, as a :class:`~cb2.moment.ResolvableMoment` — same convention as
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
            self._schedule_at(moment, lambda: action(*args, **kwargs), self.clock_id,
                              description=f"Scheduled action on {self}")

    def fork_unsynchronized(self, forked_function: Callable, args: Sequence = (), kwargs: dict = None) -> None:
        """
        Run `forked_function` on a plain background thread — *not* a child clock, and not synchronized
        to musical time. Use this for side work (I/O, GUI callbacks, etc.) that shouldn't participate
        in the scheduler. `current_clock()` is None inside it and it cannot fork child clocks, but its
        thread is tagged "unsynchronized" so it *may* still call the sleep-based wait()/wait_forever()
        — those become plain real-time sleeps (no tempo, so `units` is ignored).

        (Step 9 will route this through a reusable thread pool; for now it spawns a daemon Thread.)
        """
        _spawn_unsynchronized(forked_function, args, kwargs)

    def wait_forever(self) -> None:
        """
        Block this clock's own thread indefinitely, yielding to the scheduler so child clocks keep
        running. Typically called once a clock has forked its work and has nothing left to do itself
        (e.g. the master clock keeping the main thread alive). Returns when the clock is killed.

        Implemented as a single _wait(None): no wakeup is scheduled, so the clock simply parks until
        kill() wakes it.
        """
        try:
            self._wait(None)
        except (ClockKilledError, DeadClockError):
            pass

    def wait_for_children_to_finish(self) -> None:
        """
        Block this clock's own thread until all of its child clocks have finished, yielding to the
        scheduler so they can run, then return as soon as the last child ends.
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
        except (ClockKilledError, DeadClockError):
            pass
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
            self.wait_forever()

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

    def kill(self) -> None:
        """
        End the function running on this clock and cascade to all descendants.

        Removes any queued scheduler events for self/descendants (pending wakeups, pending forks),
        flips state to DEAD, and wakes any parked wait so it raises ClockKilledError. The whole thing
        runs under `_tree_lock` so it's atomic against a concurrent fork(): either we collect a child
        and kill it, or fork sees us DEAD and refuses — never an orphan. (A victim's wakeup that fires
        before we set DEAD is harmless: the victim sees ALIVE, advances, runs to its next wait, and
        raises DeadClockError there — the same "killed while awake" path handled below.)

        This is as good a place as any to clarify the possible code paths for terminating clocks:

            SUB-CLOCK, natural end:
                User function returns inside _fork_wrapper. The except blocks aren't entered.
                _fork_wrapper removes child from parent._children, sets state=DEAD, then notifies
                _scheduler_park_condition (the scheduler is parked there awaiting our next
                wait that will never come). Thread exits.

            SUB-CLOCK, killed:
                kill() (possibly cascaded from an ancestor) sets state=DEAD on us, removes
                our queued wake-up (and any other event related to this clock) from the heap,
                sets _wait_event, notifies _scheduler_park_condition, and detaches us from our
                parent's _children list. If we were parked inside wait() at STEP 3, we wake at
                STEP 4a, see DEAD, and raise ClockKilledError. If we weren't in wait, the next
                wait() raises DeadClockError at the entry check (STEP 1). Either error propagates
                into _fork_wrapper, which catches it and falls through to the same cleanup as the
                natural-end path (the cleanup's detach and notify are both idempotent).

            MASTER, natural end:
                Main thread finishes its top-level code. There is no _fork_wrapper wrapper, so no
                automatic notify. The scheduler would be left parked on _scheduler_park_condition,
                but since it's a daemon thread and the process is exiting, this doesn't matter.
                (Will matter once run_as_server lands and the master runs in a background
                thread — Step 5 will need its own cleanup wrapper.)

            MASTER, killed:
                kill() sets state=DEAD, removes queued wake-ups, sets _wait_event, and notifies
                _scheduler_park_condition (releasing the scheduler — critical for the master since
                there's no _fork_wrapper wrapper to do it later). If the main thread is parked inside
                wait() at STEP 3, it wakes, sees DEAD at STEP 4a, and raises ClockKilledError.
                Uncaught, the main thread dies and the daemon scheduler dies with the process —
                but for run_as_server-style scenarios where the scheduler singleton must remain
                usable for a subsequent master, the kill() notify is what makes that possible.
        """
        if self._state is ClockState.DEAD:
            return

        # Everything below runs under _tree_lock: collecting victims walks the tree, and the
        # flag-DEAD + cancel-events + detach must be atomic against a concurrent fork() (see docstring).
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

    def __repr__(self):
        child_list = "" if len(self._children) == 0 else ", ".join(str(child) for child in self._children)
        return ("Clock('{}')".format(self.name) if self.name is not None else "UNNAMED") + "[" + child_list + "]"