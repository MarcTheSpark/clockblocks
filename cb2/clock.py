import functools
import logging
import threading
from enum import Enum
from itertools import count
from numbers import Real
from typing import Callable, Sequence, Union, Iterator
from cb2.tempo_envelope import TempoHistory
from cb2.scheduler import get_scheduler, Scheduler
from cb2.metric_phase import MetricPhaseTarget
from cb2.enums import DurationUnits
from cb2.utilities import _PrintColors, current_clock
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


class ClockState(Enum):
    PENDING = "pending"  # forked, awaiting start_delay; thread not yet running user code
    ALIVE = "alive"      # forked_function is running (or about to run)
    DEAD = "dead"        # killed, or forked_function returned


def _reschedule_after_tempo_change(fn):
    """
    Decorator for Clock tempo/rate/beat_length setters. Holds the scheduler, brings the target clock's
    tempo_history up to the current scheduler position (so the change applies forward, not retroactively
    from the last committed beat), applies the mutation, then recomputes the scheduler-time of any queued
    wakeups whose `acting_clock` is self or a descendant.
    """
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with self.scheduler.held():
            self.bring_up_to_date()
            result = fn(self, *args, **kwargs)
            self._reschedule_self_and_descendants()
        return result
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

        if self.is_master():
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
        descendant of self has a stale `t` (computed against the old tempo curve). Recompute via
        clock_to_scheduler_time(target_beat) for each match.
        """
        affected = {id(c) for c in self.iterate_descendants(include_self=True)}

        def matches(event):
            meta = event.metadata
            if not isinstance(meta, dict):
                return False
            ac = meta.get("acting_clock")
            return ac is not None and id(ac) in affected and "target_beat" in meta

        def recompute(event):
            meta = event.metadata
            return meta["acting_clock"].clock_to_scheduler_time(meta["target_beat"])

        self.scheduler.reschedule(matches, recompute)

    ##################################################################################################################
    #                                              Waiting and Forking
    ##################################################################################################################

    def wait(self, dt, units="beats"):
        """
        Block this clock for `dt` beats (or seconds, if units="time"), yielding to the scheduler.

        Implemented in four steps:
            (1) reject the call if this clock isn't ALIVE, or if we're not on this clock's own thread;
            (2) compute the wake-up time and queue a wake-up event on the scheduler;
            (3) hand off to the scheduler and block;
            (4) clean up on wake — raising ClockKilledError if we were killed mid-wait, otherwise advancing the
                committed pointer in tempo_history to reflect the post-wait beat/time.

        See in-line STEP comments for details and see the `kill()` docstring for the full picture
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

        # ------------------- STEP 2: Calculate and schedule wake up in scheduler --------------------

        units = DurationUnits(units)
        if units == DurationUnits.BEATS:
            wake_up_beat = self.beat() + dt
            wake_up_time = self.tempo_history.time_at_beat(wake_up_beat)
        else:
            wake_up_time = self.time() + dt
            wake_up_beat = self.tempo_history.beat_at_time(wake_up_time)
        wake_up_time_in_scheduler = self.clock_to_scheduler_time(wake_up_time, units="time")

        # queue the wake-up event with the scheduler
        self.scheduler.schedule_action(
            wake_up_time_in_scheduler,
            self._wake_and_advance_to_next_wait_call,
            self.clock_id,
            {"description": f"{self} wakeup action",
             "acting_clock": self,
             "target_beat": wake_up_beat}
        )

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

        # ~~~~~ Step 4b: Update self.tempo_history to reflect new time post-wait ~~~~~
        # advance the committed pointer of tempo_history from where it was to where we just woke up.
        # Use tempo_history.beat() directly (not self.beat(), which is the live scheduler-derived position).
        self.tempo_history.advance(wake_up_beat - self.tempo_history.beat())

    def _wake_and_advance_to_next_wait_call(self):
        """
        This function is run on the scheduler thread: it wakes up this clock, and then blocks until the clock
        has hit another wait call and scheduled its next wakeup.
        """
        with self._scheduler_park_condition:
            self._wait_event.set()
            self._scheduler_park_condition.wait()

    def _resolve_start_delay(self, schedule_at: Union[float, MetricPhaseTarget, None]) -> float:
        """
        Translate fork()'s `schedule_at` into a delay, in this (the parent) clock's beats, from now.

            None              -> 0 (start immediately)
            a beat number     -> that beat minus the current beat (clamped to 0 if it's in the past)
            MetricPhaseTarget -> the delay to the next beat matching the requested phase
        """
        if schedule_at is None:
            return 0
        if isinstance(schedule_at, Real):
            start_delay = schedule_at - self.beat()
            if start_delay < 0:
                logging.warning("`schedule_at` argument specified a beat in the past; forking immediately.")
                return 0
            return start_delay
        if isinstance(schedule_at, MetricPhaseTarget):
            # get_nearest_matching_beats returns the nearest match below and above, in order of nearness;
            # we want the match above, since it's in the future, so we use max
            return max(*schedule_at.get_nearest_matching_beats(self.beat())) - self.beat()
        raise ValueError("`schedule_at` must be either a float, a MetricPhaseTarget, or None")

    def fork(self, forked_function: Callable, args: Sequence = (), kwargs: dict = None, name: str = None,
             initial_rate: float = None, initial_tempo: float = None, initial_beat_length: float = None,
             schedule_at: Union[float, MetricPhaseTarget] = None, done_callback: Callable[[], None] = None):
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
        :param schedule_at: either a beat or a :class:`~clockblocks.tempo_envelope.MetricPhaseTarget` specifying when we
            want the forked function to begin. The default value of None indicates that it is to begin immediately. A
            float indicates the beat in this clock at which the forked function is to start (should be in the future).
            Alternatively, a MetricPhaseTarget can be used to specify where in a regular cycle it should begin.
            For instance, if we want to sync every fork to 3/4 time, MetricPhaseTarget(0, 3) would start it on
            the downbeat, MetricPhaseTarget(1, 3) would start it on beat 2, and MetricPhaseTarget(2.5, 3) would start it
            halfway through beat 3.
        :param done_callback: a callback function to be invoked when the clock has terminated
        :return: the spawned child clock
        """
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

        start_delay = self._resolve_start_delay(schedule_at)

        # ------------------- STEP 3: Define the child's lifecycle wrapper (_fork_wrapper) -------------------

        def _fork_wrapper(*args, **kwds):
            # ~~~~~ Wrapper step 1: Thread/clock setup ~~~~~
            # Bind __clock__ so current_clock() resolves to the child inside the user function,
            # finalize parent_offset now that start_delay has elapsed, and flip ALIVE.
            threading.current_thread().__clock__ = child
            child.parent_offset += start_delay
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
            # Detach from the forking parent (self) and mark DEAD. This code is for natural clock exits
            # and is redundant for killed clocks. (The if/in check exists for the killed case, since
            # kill already removed the child and trying to remove again would cause a ValueError)
            if child in self._children:
                self._children.remove(child)
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

        self.scheduler.schedule_action(
            self.clock_to_scheduler_time(self.beat() + start_delay),
            _start_new_clock,
            child.clock_id,
            {"description": f"Forking of {child}",
             "acting_clock": self,
             "forked_child": child,
             "target_beat": self.beat() + start_delay}
        )
        return child

    @property
    def alive(self) -> bool:
        """True if this clock is currently running (PENDING and DEAD both return False)."""
        return self._state is ClockState.ALIVE

    def kill(self) -> None:
        """
        End the function running on this clock and cascade to all descendants.

        Removes any queued scheduler events for self/descendants (pending wakeups, pending forks),
        flips state to DEAD, and wakes any parked wait so it raises ClockKilledError. The scheduler
        is briefly held while we mutate the heap to avoid racing a wakeup that's about to fire.

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

        with self.scheduler.held():
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
            # killed during start_delay never has _fork_wrapper run at all).
            if c.parent is not None and c in c.parent._children:
                c.parent._children.remove(c)

    def __repr__(self):
        child_list = "" if len(self._children) == 0 else ", ".join(str(child) for child in self._children)
        return ("Clock('{}')".format(self.name) if self.name is not None else "UNNAMED") + "[" + child_list + "]"