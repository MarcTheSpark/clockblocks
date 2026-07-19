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
Module containing the :class:`Scheduler`, the single background thread that drives an entire clock family:
it holds a queue of upcoming events, sleeps until the next one is due, and wakes the clocks waiting on them.
Also contains the :class:`TimingBackend` that supplies the scheduler's notion of "now" and how it sleeps,
along with :class:`CompressedTime`, a backend that makes a clock family run in scaled-down real time.
(Currently used for testing purposes.)
"""

import threading
import time
import heapq
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class TimingBackend:
    """
    The scheduler's pluggable source of "now" and of the condition it uses for waiting.

    The scheduler (and therefore an entire clock family) has exactly one timed sleep — the run loop's
    wait on ``_queue_change_condition`` (see :meth:`Scheduler.run` STEP 1c). That condition is sourced from
    the TimingBackend passed in the constructor, and defaults to a regular condition on a plain lock.
    Similarly, all timing reads within the scheduler run through timing_backend.now().
    The backend therefore defines both how time is perceived and how the scheduler waits.

    See :class:`CompressedTime` for a backend used to run fast deterministic tests.
    """

    def now(self) -> float:
        """
        The backend's current time, in seconds. Only differences between readings are meaningful;
        the origin is arbitrary.
        """
        return time.perf_counter()

    def get_sleep_condition(self) -> threading.Condition:
        """
        A fresh condition variable for the scheduler's run loop to wait on. Timeouts passed to its
        ``wait()`` are interpreted in the same units :meth:`now` returns.
        """
        return threading.Condition(threading.Lock())


def _default_time_backend() -> 'TimingBackend':
    """The backend a :class:`Scheduler` uses when none is passed. The test suite overrides this module
    attribute to inject a :class:`CompressedTime` family-wide (see ``tests/timing.py``)."""
    return TimingBackend()


class CompressedTime(TimingBackend):
    """
    A :class:`TimingBackend` that runs scaled real time: :meth:`now` advances ``factor`` times faster
    and the sleep condition's ``wait()`` divides its timeout by ``factor``. A clock family on this backend
    runs deterministically fast while cross-thread handshakes still happen as usual.

    Caveat: handshake / context-switch overhead is *not* compressed, so a T-second wait costs roughly
    ``T / factor + handshake_overhead``. Very large factors hit that floor and loosen achievable timing
    tolerances (see ``tests/README.md``). 10x is a good default.
    """

    def __init__(self, factor: float = 10.0):
        if factor <= 0:
            raise ValueError("Compression factor must be positive.")
        self.factor = factor
        self._t0 = time.perf_counter()

    def now(self) -> float:
        """Real time since this backend was created, scaled up by ``factor``."""
        return (time.perf_counter() - self._t0) * self.factor + self._t0

    def get_sleep_condition(self) -> threading.Condition:
        """A condition whose ``wait()`` shortens any timeout by ``factor``, so that sleeps in
        compressed time take correspondingly less real time."""
        factor = self.factor

        class _CompressedCondition(threading.Condition):
            def wait(self, timeout=None):
                return super().wait(None if timeout is None else timeout / factor)

        return _CompressedCondition(threading.Lock())


@dataclass(order=True)
class QueueEvent:
    # Ordering is by t first, then by priority. action/metadata are not orderable.
    t: float
    priority: Tuple[int, ...]
    action: Callable[[], None] = field(compare=False)
    metadata: Any = field(default=None, compare=False)


class Scheduler(threading.Thread):
    """
    The background timing thread for one clock family. Internal: user code need never touch it directly.
    Each master clock mints and owns its own Scheduler, children share their master's, and the public
    timing knobs are surfaced on :class:`~clockblocks.clock.Clock`
    (e.g. :attr:`~clockblocks.clock.Clock.timing_policy`, fast-forward).

    Runs a single loop over a time-ordered heap of events, sleeping until the next is due and then
    firing it. Utilizes two locks for concurrency safety: :attr:`_queue_change_condition`, for anything
    that touches the event Queue, and :attr:`_execution_lock` which is held throughout an event's action.
    See inline commentary in :meth:`run`.

    In the clock system each event wakes a clock and parks until the user code runs up
    to its next ``wait()``. Tempo changes reschedule pending events using :meth:`reschedule`,
    and killing a clock removes them via :meth:`remove_events`.

    :param timing_policy:
        Bounds how far each wait may deviate from its nominal length in order to track the absolute
        schedule. 0 -> Absolute timing (cut/extend the wait as needed to match ideal time since start).
        1 -> Relative timing (wait exactly the nominal delay, drift never corrected). 0.98 (default) ->
        a wait may be compressed to 98% (or stretched to 102%) of its nominal length to correct any
        accumulated drift.
    :param precise_timing:
        When True, close the final approach to each event with a busy-spin instead of resting on
        the (jittery) OS wait timeout, hitting the event time to within microseconds. Costs one core
        for at most ``spin_guard_duration`` seconds per event — see :meth:`run` STEP 1c.
    :param spin_guard_duration:
        Width (seconds) of the busy-spin guard band used when ``precise_timing`` is on. The coarse
        OS wait stops this far short of the deadline and the remainder is spun out. Default 500µs.
    :param daemon: run as a daemon thread (the default) so it can't keep the process alive on its own.
    :param time_backend:
        The :class:`TimingBackend` supplying "now" and the sleep condition. Defaults to real
        ``perf_counter`` time; tests pass :class:`CompressedTime` to run faster.
    """

    def __init__(self, timing_policy: float = 0.98, precise_timing: bool = False,
                 spin_guard_duration: float = 0.0005, daemon: bool = True,
                 time_backend: 'TimingBackend | None' = None):
        super().__init__(daemon=daemon)
        self.timing_policy = timing_policy
        self.precise_timing = precise_timing
        self.spin_guard_duration = spin_guard_duration
        self._time = time_backend or _default_time_backend()

        # Heap-based queue for scheduled events.
        self._queue = []

        # Two synchronization primitives, kept deliberately separate. How each is used (and why) lives
        # where it matters — see run() for _queue_change_condition and held() for _execution_lock.
        #   _queue_change_condition — guards the heap and signals changes to it: anything that reads or
        #       writes _queue holds it, and notify_all() wakes the run loop to re-evaluate the head.
        #   _execution_lock — held by the run loop for the full duration of each action's execution, so
        #       "locked" == "mid-action". held() takes it to mutate action-related state safely.
        # Lock order: the run loop never holds _queue_change_condition while taking _execution_lock — if
        # it did, a running action couldn't modify the queue. For the clock system that's essential:
        # actions routinely schedule wake-ups, fork, and reschedule after tempo changes.
        #
        # The default condition (see TimingBackend) uses a plain (non-reentrant) Lock on purpose: every
        # critical section here is flat, so accidental reentrancy would signal a bug and should deadlock
        # loudly rather than be hidden by Condition's default RLock. We don't need to keep a reference to
        # the lock object, since `with self._queue_change_condition:` acquires it directly.
        self._queue_change_condition = self._time.get_sleep_condition()
        self._execution_lock = threading.Lock()
        self._killed = False

        # Timing variables.
        self._start_time = None
        self._last_event_time = None
        self._ideal_time = 0.0  # Ideal scheduler time (seconds since start)

        # _fast_forward_goal should be one of the following:
        #  1) None, meaning that we are not fast-forwarding; the normal state of the scheduler
        #  2) A scheduler time up until which we process events as fast as possible without sleeping, advancing
        #   _ideal_time as we go.
        #  3) float('inf), indicating that events will be processed as fast as possible until the fast forwarding
        #   state is turned off
        self._fast_forward_goal = None

        # When fast-forward ends we need to re-anchor the timing reference points (see _reanchor_timing), since
        # otherwise we would appear to be far ahead of schedule. There are two ways that fast-forward can end:
        #   (1) We reach a finite _fast_forward_goal. This is detected within _end_fast_forward_if_active when the
        #       next event lands at/beyond the goal, and _reanchor_timing is called directly. In this case the 
        #       _was_fast_forwarding flag below isn't used for detection, but it still must be cleared, because
        #       otherwise the next normal wait would think we had just been fast-forwarding and reanchor spuriously.
        #   (2) Fast forward was actively turned off (i.e. _fast_forward_goal was set back to None before it was
        #       reached). In this case, we need to know on the next run loop that we *were* fast_forwarding so
        #       that we can call _reanchor_timing. This is accomplished through the _was_fast_forwarding flag.
        #       Without this flag we would have no way of distinguishing between fast-forwarding having always
        #       been off, and fast-forwarding having just been turned off.
        # See _fast_forwarding_through (which sets the flag) and _end_fast_forward_if_active (which consumes it).
        self._was_fast_forwarding = False

    def time(self) -> float:
        """Return the scheduler's ideal time (the time that should have passed by schedule).

        This is *event-quantized*, not wall-clock-interpolated: it is bumped to each event's scheduled time
        as that event executes (see :meth:`_execute_event`), so between events it holds the most recently
        executed event's time. See :meth:`~clockblocks.clock.Clock.time` for what that means for reads taken
        between events / from other threads. For a smoothly-advancing estimate of the position *between* events,
        see :meth:`projected_time`."""
        return self._ideal_time

    def projected_time(self) -> float:
        """A wall-clock-interpolated estimate of where the scheduler is *right now*, as opposed to :meth:`time`,
        which returns the time of the last executed event.

        Between events, :meth:`time` stays put while real time passes; this fills the gap by interpolating from
        the last event toward the next, so continuous readers (e.g. parameter automation) see smooth motion. The
        estimate depends on :attr:`timing_policy`, which decides when the next event is planned to arrive.

        This is an estimate, and not monotonic. It resets on deliberate schedule changes (a fast-forward, a
        live `timing_policy` retune), and booking an event below the current projection pulls it back to that
        event. (This kind of makes sense, since the free-climb overshot a point where something needed to happen.)
        In both cases the cause of the backward step is that we are collapsing the gap between a stale committed
        time and an overly optimistic projected time.

        To avoid this issue entirely, use the :meth:`held` context manager, which first rouses
        the scheduler to the projected position.

        Refer to ``diagrams/projectedTimeExplanation`` for how this resolves under the various scenarios (ahead
        of schedule, behind, empty queue, etc.).
        """
        if self._last_event_time is None or self._fast_forward_goal is not None:
            # if fast-forwarding, there's no concept of progress between events, so the only logical projected
            # time is ideal_time. If last_event_time is None, timing hasn't been anchored yet (nothing has
            # required a real wait — see run()), so time isn't passing: projected time is just self._ideal_time.
            return self._ideal_time

        try:
            # list access is atomic, so this one-liner will never give us something malformed even though it's
            # reading from a (possibly in the middle of being operated on) event queue. Note that checking if
            # len(self._queue) > 0 and then accessing if not is dangerous, because it should get popped in between
            # so instead we check for an empty queue by catching an index error
            next_event = self._queue[0]
        except IndexError:
            # empty queue: no next event to interpolate toward, so climb freely at real-time rate based
            # on the last ideal time / wall time
            return self._ideal_time + (self._time.now() - self._last_event_time)

        ideal_wait_duration = next_event.t - self._ideal_time
        if ideal_wait_duration <= 0:
            # degenerate case in which a next event has been scheduled at or before the already commited ideal time
            # this could happen naturally if we're mid-execution of contemporaneous events, or if someone has scheduled
            # something in the past and the (immediate) execution of that event hasn't happened yet
            return self._ideal_time

        planned_wall_duration = self._target_wall_time(next_event) - self._last_event_time
        if planned_wall_duration <= 0:
            # degenerate case that can only happen in an absolute timing policy where we're already past due
            # (in this case, _target_wall_time will return _last_event_time, requesting immediate wake)
            # since the next event will be fired imminently, return its time
            return next_event.t

        # if we reach this point in the code, there is a coherent, positive planned wall duration
        # So we simply measure our progress by comparing how long we have waited since the last event
        # to the planned wait (which incorporates the timing policy), Then we project that progress onto
        # the absolute schedule. (Note that we clamp progress between 0 and 1.)
        progress_to_next_event = (self._time.now() - self._last_event_time) / planned_wall_duration
        return self._ideal_time + ideal_wait_duration * min(max(progress_to_next_event, 0.0), 1.0)

    def wall_time(self) -> float:
        """
        Return the actual time elapsed on the schedule. Returns 0 until timing is anchored by the first
        event that requires a real wait (see :meth:`run`), then advances in real time. Also resets to ideal
        time after fast-forwarding, erasing any prior lag.

        Measured through the :class:`TimingBackend`, which defaults to ``time.perf_counter``. `perf_counter` is
        monotonic and high-resolution, so this is true elapsed real time, never affected by NTP corrections
        (slews or steps) to CLOCK_REALTIME. Under a compressed backend it is scaled in step with everything else.
        """
        return self._time.now() - self._start_time if self._start_time else 0.0

    def lag(self) -> float:
        """
        How far behind the absolute schedule the scheduler is currently running, in seconds: the real time
        that has elapsed minus the ideal time that should have elapsed.

        Positive means events are firing late. It grows when an event's action takes longer than the gap to
        the next event, and (under a relative timing policy) is allowed to persist rather than being chased
        down. Fast-forwarding resets it to zero, since :meth:`_reanchor_timing` re-anchors ``_start_time``
        to put us exactly on the absolute schedule.
        """
        return self.wall_time() - self.time()

    def set_fast_forward_goal(self, goal: float | None) -> None:
        """
        Set the fast-forward goal (a scheduler-time, float('inf') for indefinite, or None to stop) and
        wake the run loop so it acts on the change immediately rather than after its current timed wait.
        """
        # Mutated under _queue_change_condition to avoid interleaving with the run loop.
        with self._queue_change_condition:
            self._fast_forward_goal = goal
            self._queue_change_condition.notify_all()

    def is_fast_forwarding(self) -> bool:
        """Whether a fast-forward goal is currently set."""
        return self._fast_forward_goal is not None

    def kill(self) -> None:
        """Stop the scheduler."""
        self._killed = True
        # Wake the run loop wherever it's parked on the condition (empty-queue wait or timed wait)
        # so it observes _killed and exits.
        with self._queue_change_condition:
            self._queue_change_condition.notify_all()

    @contextmanager
    def held(self):
        """
        Context manager that: 1) ensures that the scheduler is not executing scheduled actions — waiting if one is
        in flight, then preventing others from starting; 2) rouses the scheduler and updates its committed time to
        the current projected time. Code under this context manager therefore acts like an immediately scheduled
        action, holding exclusive access to an up-to-date scheduler.

        Implemented by taking `_execution_lock`, which the run loop holds for the full duration of each
        action. So don't call this from within a scheduled action: you'd block waiting for that action
        to finish, but it can't finish while it's stuck waiting here.

        In the context of the clock system, this context manager should be used any time you want to
        take clock-related actions from a non-clock thread. However, since it's implemented by taking the
        same `_execution_lock` held during scheduled actions, it should never be used *within* a scheduled
        action: the action would block waiting for itself to finish. For this reason we have
        :meth:`~clockblocks.clock.Clock.hold_scheduler`, which safely wraps this method by detecting whether we
        are running from a clock and no-op'ing in that case.
        """
        with self._execution_lock:
            self._rouse_to_now()
            yield self

    def _rouse_to_now(self) -> None:
        """
        Advance the committed position (:meth:`time`) to the current interpolated position, as if a
        zero-duration event had just fired now. Internal helper of :meth:`held` (never call it directly —
        it must run under `_execution_lock`) so that a scheduler woken between actions sees an up-to-date time.
        """
        if self._last_event_time is None or self._fast_forward_goal is not None:
            # not started, or fast-forwarding (no meaningful "now" between events): nothing to advance
            return
        # Safe by construction, because :meth:`projected_time` always lies in ``[_ideal_time, next_event.t]``:
        # the bump only ever moves ``_ideal_time`` forward (committed time stays monotonic) and never past the
        # next queued event (so nothing is skipped or mis-stamped when the run loop resumes). ``_start_time`` is
        # left untouched, so the absolute schedule — and therefore the next event's wall-clock target — is
        # preserved; we re-anchor ``_last_event_time`` to now, exactly as a real event would.
        #
        # The write is under _queue_change_condition because this method runs on a *foreign* thread, and the run loop
        # reads this pair together in Step 1b (under _compute_wait_duration -> _target_wall_time) under that same lock.
        # Without it, run-loop STEP 1 could read a torn pair — new _ideal_time against the old _last_event_time — and
        # compute the wrong wait. Lock order is _execution_lock (already held) → _queue_change_condition, matching
        # _execute_event.
        with self._queue_change_condition:
            self._ideal_time = self.projected_time()
            self._last_event_time = self._time.now()

    def remove_events(self, matches: Callable[['QueueEvent'], bool]) -> int:
        """
        Remove all queued events that satisfy `matches(event)`. Returns the number removed.
        Used when a clock is killed and its pending wakeups (and pending forks) need to be cancelled.
        """
        with self._queue_change_condition:
            kept = [e for e in self._queue if not matches(e)]
            removed = len(self._queue) - len(kept)
            if removed:
                self._queue = kept
                heapq.heapify(self._queue)
                self._queue_change_condition.notify_all()
        return removed

    def reschedule(self, matches: Callable[['QueueEvent'], bool],
                   recompute: Callable[['QueueEvent'], float]) -> None:
        """
        Walk the heap and recompute `t` for any event that satisfies `matches(event)`.
        Used when a tempo change makes previously-scheduled wakeups stale.
        """
        with self._queue_change_condition:
            changed = False
            for i, event in enumerate(self._queue):
                if matches(event):
                    new_t = recompute(event)
                    if new_t != event.t:
                        self._queue[i] = QueueEvent(new_t, event.priority, event.action, event.metadata)
                        changed = True
            if changed:
                heapq.heapify(self._queue)
                self._queue_change_condition.notify_all()

    def schedule_action(self, t: float, action: Callable, priority: Tuple[int, ...] = (0,), metadata: Any = None) -> None:
        """
        Schedule the given action to be executed at time 't' (in seconds since scheduler start).
        """
        event = QueueEvent(t, priority, action, metadata)
        with self._queue_change_condition:
            heapq.heappush(self._queue, event)
            self._queue_change_condition.notify_all()
        logger.debug("Scheduled event %r for time %s", metadata, t)

    def run(self) -> None:
        """
        Main scheduler loop: wait until the next queued action is due, then execute it, forever.

        The two subtle parts — how the wait is structured around `_queue_change_condition` so a queue
        change can't be lost, and why execution is wrapped in `_execution_lock` — are explained inline
        at STEP 1 and STEP 2 below.

        Note that the clock starts in an unanchored state (`_start_time = _last_event_time = None`)
        This is intentional: we don't want the code that runs before the first future action is scheduled
        (which can be lengthy initial setup) to cause us to start off way behind schedule. The timing is
        anchored (via `_reanchor_timing`) when the first future event is pulled from the queue.
        """
        logger.info("Scheduler started")
        while not self._killed:
            # -------------------------   STEP 1: Wait for the next event -------------------------------
            # We hold the _queue_change_condition throughout so that nothing modifies the queue while we are
            # reading from it and calculating the wait time. Note that, since this is a Condition,
            # the underlying lock is released during self._queue_change_condition.wait(), allowing methods like
            # schedule_action, reschedule, and remove_events to modify the queue while we are waiting in between
            # events. Those methods then notify_all() on the condition, which wakes us from wait() to re-check
            # the now-updated queue.

            with self._queue_change_condition:
                # ~~~~~~ STEP 1a: If the queue is empty, wait for an event to be scheduled ~~~~~~
                # This wait is for when the scheduler is alive and nothing is scheduled. So we wait to be notified
                # that there has been some change to the _queue. Note that kill() also notifies _queue_change_condition,
                # which wakes us from this wait() and then breaks us out of the run loop after seeing self._killed=True
                while not self._queue and not self._killed:
                    self._queue_change_condition.wait()

                if self._killed:
                    break

                # ~~~~~~ STEP 1b: Peek at the queue, and calculate wait duration to next scheduled event ~~~~~~
                next_event = self._queue[0]
                now = self._time.now()

                # Fast-forward short-circuits the wait. If we're fast-forwarding *through* this event (it
                # lies before the goal), fire it with no wall-clock delay. Otherwise we're about to wait in
                # real time, so first settle any fast-forward that was in progress — re-anchoring the timing
                # reference points — and then time the wait normally.
                if self._fast_forwarding_through(next_event):
                    self._was_fast_forwarding = True
                    wait_duration = 0.0
                else:
                    self._end_fast_forward_if_active(now)
                    if self._start_time is None and next_event.t <= self._ideal_time:
                        # Timing is not yet anchored, and this event is immediate (e.g. a clock's
                        # initial wake, scheduled at the current ideal time): fire it with no wait
                        # and leave timing unanchored, so that setup time stays off the schedule.
                        wait_duration = 0.0
                    else:
                        if self._start_time is None:
                            # First event that requires a real wait: plant the wall-clock anchor
                            # here, so the wait is measured in full from this instant.
                            self._reanchor_timing(now)
                        wait_duration = self._compute_wait_duration(next_event, now)

                # ~~~~~~ STEP 1c: Wait for the next scheduled event (if in the future) ~~~~~~
                # Two paths, depending on whether we're spinning (busy-waiting) to hit the target time precisely
                # Coarse mode leaves spin_deadline as None to signal no busy wait; precise mode waits coarsely
                # up to the guard band defined by spin_guard_duration, and then sets spin_deadline to
                # busy wait the final microseconds.
                spin_deadline = None
                if not self.precise_timing:
                    # Coarse mode: rest on the OS wait for the whole remaining duration, then re-enter the
                    # loop to re-evaluate. If the event is already due (<= 0), fall through to execute it now.
                    # If the event is still not quite due, we wait again on the tiny remaining duration until
                    # it eventually passes the deadline.
                    if wait_duration > 0:
                        self._queue_change_condition.wait(timeout=wait_duration)
                        continue
                else:
                    # Precise mode: coarse-wait down to within spin_guard_duration of the deadline
                    # (as above this might take a couple passes), then busy-spin the remainder below
                    # for a microsecond-accurate landing.
                    if wait_duration > self.spin_guard_duration:
                        self._queue_change_condition.wait(timeout=wait_duration - self.spin_guard_duration)
                        continue
                    elif wait_duration > 0:
                        # Inside the guard band, but not past the deadline, so we arm the busy wait.
                        # The deadline is measured from `now` (the same time the wait_duration was computed
                        # against) so it is exactly the intended wake time.
                        # Note that we don't spin *here* because we're holding _queue_change_condition's
                        # lock. The scheduler is effectively unresponsive while holding that lock, since
                        # anything that modifies the queue needs it. During any of the _queue_change_condition.wait
                        # calls above, this is not an issue, because the lock is actually released while
                        # waiting on a condition; that's part of how conditions work.
                        spin_deadline = now + wait_duration
                    # else (<= 0): already due — fall through to execute now.

            # Having exited the with `self._queue_change_condition` block and released the lock, we now check
            # if spin_deadline is set (i.e. if we're using precise_timing and within the guard band)
            if spin_deadline is not None:
                while self._time.now() < spin_deadline and not self._killed:
                    # We poll _killed while spinning so a kill() during the spin tears the loop down promptly.
                    # Since _queue_change_condition was released, other threads are free to modify the queue here;
                    # however, we won't respond to those changes until after the (very very short) busy wait
                    pass
                if self._killed:
                    break

            # NOTE: there is a gap here between waiting for the event time and performing the action, where we are
            # no longer holding _queue_change_condition. This leaves a window for an external thread to mutate the
            # queue, perhaps even preempting the event we're waiting for with an earlier event. For this reason
            # we recheck validity at the start of _execute_event and bail if things changed.

            # --------------------------- STEP 2: Perform the scheduled action -------------------------------
            # Executed events happen under the _execution_lock. If an external thread wants to make sure that
            # we are not mid-action, it can use the context manager held().
            #
            # In the context of the Clock system, the scheduled action parks the scheduler and allows the clock
            # to perform user code until the next wait. clock._reschedule_after_tempo_change uses held()
            # to ensure that we only modify tempo/reschedule events once every clock is dormant again.
            with self._execution_lock:
                self._execute_event(next_event)
        logger.info("Scheduler terminated")

    def _fast_forwarding_through(self, next_event: 'QueueEvent') -> bool:
        """
        Is a fast-forward in effect and does this event fall before the goal, so it should
        fire with no wall-clock wait?

        A None goal will always return False; we're not fast-forwarding. A float('inf') goal will always
        return True. Any other finite goal will return True if the next event is before the _fast_forward_goal.
        """
        return self._fast_forward_goal is not None and next_event.t < self._fast_forward_goal

    def _end_fast_forward_if_active(self, now: float) -> None:
        """
        Checks for and handles the transition from fast-forwarding to non-fast-forwarding state.
        Called from the run loop (while holding _queue_change_condition) whenever we are not fast_forwarding
        through the next event. Generally a no-op if we are not and have not been fast forwarding. The two
        cases where this function acts are:

          (1) When we have reached a finite _fast_forward_goal; in this case is_fast_forwarding() is still true,
            so we pin _ideal_time to the goal, turn off fast-forwarding (by setting _fast_forward_goal = None),
            clear _was_fast_forwarding (so that the next wait doesn't think we need to re-anchor), and
            _reanchor_timing to the current time.

          (2) When fast forwarding has been turned off externally. In this case, there's no _fast_forward_goal
            to read from or flip, so we're relying on _was_fast_forwarding. Flip that to false and _reanchor_timing.
        """
        if self.is_fast_forwarding():
            # we are fast-forwarding, but since this function was called, it means that we're about to stop
            # since the next event is set for *after* the _fast_forward_goal. Jump the _ideal_time straight
            # to the fast forward goal (making sure it's not backwards), clear the _fast_forward_goal,
            # and then reanchor timing
            self._ideal_time = max(self._ideal_time, self._fast_forward_goal)
            self._fast_forward_goal = None
            self._reanchor_timing(now)
            self._was_fast_forwarding = False
            return
        elif self._was_fast_forwarding:
            # We're not fast forwarding anymore, but last run iteration we were. This branch is only reached
            # when fast forwarding was canceled manually (by clearing self._fast_forward_goal); otherwise
            # we would have exited fast-forward via the block above.
            # in this case, all we need to do is reanchor the timing
            self._reanchor_timing(now)
            self._was_fast_forwarding = False
            return
        # if we neither are fast-forwarding nor were fast-forwarding, this is a normal non-fast-forwarding wait
        # return naturally as a no op

    def _reanchor_timing(self, now: float) -> None:
        """
        (Re)anchor the timing reference points — used to plant the initial anchor at the first real wait
        (see run()) and to re-anchor after fast-forwarding:
            - _last_event_time becomes now, so the next wait measures relative timing from this instant.
            - self._start_time is set to self._ideal_time seconds in the past so that we are anchored
                to be exactly on time as far as absolute timing is concerned.
        """
        self._last_event_time = now
        self._start_time = now - self._ideal_time

    def _target_wall_time(self, next_event: 'QueueEvent') -> float:
        """The wall instant at which we plan to arrive at `next_event` (i.e. at scheduler time
        ``next_event.t``), given the current timing policy. This is the single, shared definition of "when do we
        intend to fire the next event": :meth:`_compute_wait_duration` subtracts `now` from it to get a wait,
        and :meth:`projected_time` interpolates against it to estimate the current position.

        This method never touches the queue directly; instead, it assumes that its callers have done so
        responsibly in order to pass next_event. Recomputes from live state on every call and therefore
        robust to a changing schedule.
        """
        # the time between the last event and this next event in ideal, scheduler time
        ideal_wait_duration = next_event.t - self._ideal_time
        # absolute_target = the target wall time that would land exactly on the absolute schedule
        # (take the wall time when the scheduler started, and add the ideal, scheduled time of the next event)
        absolute_target = self._start_time + next_event.t
        if ideal_wait_duration <= 0:
            # Degenerate case in which the next event is scheduled in the past, *before* scheduled time
            # of the last event that fired. In this case, relative timing makes no sense so we should aim
            # to get back on the absolute schedule.
            return absolute_target
        else:
            # The timing policy sets an allowable deviation from the ideal relative wait time.
            # We take the absolute target wall time (tracking absolute time since start) and clamp it
            # on the low side by `self._last_event_time + ideal_wait_duration * self.timing_policy`,
            # and on the high side by `self._last_event_time + ideal_wait_duration / self.timing_policy`
            # (or inf if self.timing_policy is 0).
            #
            # This is easiest to understand with a few examples:
            #
            # - if timing policy is 0, the clamp is (self._last_event_time, ∞), essentially a no-op, leading us
            # to use the absolute target. (If that target is *before* the last event, technically this clamps to
            # the last event time, but that leads to a wait of 0 regardless.)
            # - if timing policy is 1, the clamp is (self._last_event_time + ideal_wait_duration,
            # self._last_event_time + ideal_wait_duration), which simply insists upon the relative wait.
            # - if timing policy is 0.5, the clamp is (self._last_event_time + ideal_wait_duration * 0.5,
            # self._last_event_time + ideal_wait_duration * 2) meaning that we can fudge the time between
            # events as low as half as long (or as much as much as twice as long) as scheduled to get back on track.
            lo = self._last_event_time + ideal_wait_duration * self.timing_policy
            hi = (self._last_event_time + ideal_wait_duration / self.timing_policy) \
                if self.timing_policy > 0 else float("inf")
            return min(hi, max(lo, absolute_target))

    def _compute_wait_duration(self, next_event: 'QueueEvent', now: float) -> float:
        """Calculate the correct wait duration based on the time of the next event, the current now, and the
        timing policy."""
        target = self._target_wall_time(next_event)
        self._log_event_timing(next_event, now, target)
        # wait is simply the difference between target time and now
        return target - now

    def _log_event_timing(self, event: 'QueueEvent', now: float, target: float) -> None:
        """Detailed timing trace for one scheduled event (only logged if DEBUG is enabled).
        The wall instants (last_event_actual, now, and the three targets) are printed relative to
        _start_time — i.e. in seconds since the scheduler started — so that they can be meaningfully
        compared to the scheduler-time fields (last_event_scheduled, next_event_scheduled)."""
        if not logger.isEnabledFor(logging.DEBUG):
            return
        s = self._start_time
        # absolute_target lands exactly on the absolute schedule; relative_target lands exactly
        # ideal_wait_duration after the last event *actually* fired (drift never corrected). The policy picks
        # `target` between them. Both are recomputed here purely for the trace.
        absolute_target = self._start_time + event.t
        relative_target = self._last_event_time + (event.t - self._ideal_time)
        logger.debug(
            "event=%r last_event_scheduled=%.6f last_event_actual=%.6f next_event_scheduled=%.6f now=%.6f "
            "rel_target=%.6f abs_target=%.6f target_based_on_policy=%.6f calculated_wait=%.6f",
            event.metadata, self._ideal_time, self._last_event_time - s, event.t, now - s,
            relative_target - s, absolute_target - s, target - s, target - now,
        )

    def _execute_event(self, event: QueueEvent) -> None:
        """
        Remove the event from the queue and execute its action.
        Updates the scheduler's timing metrics accordingly.
        """
        # Confirm the head is still `event` before popping: between run()'s peek at the queue (Step 1b) and now,
        # the lock was released, so another thread may have pushed a smaller-t event or rescheduled one to the
        # front. If so, bail and let run() recompute timing against the new head — otherwise we'd pop the wrong
        # event and execute `event.action` with stale bookkeeping. We use the condition here purely as the queue
        # lock (no wait/notify), held only to pop.
        # The scheduled action then runs unguarded by _queue_change_condition; the mechanism for avoiding racing with
        # the action is to use scheduler.held(), since _execution_lock is held around this whole call.
        with self._queue_change_condition:
            if self._queue and self._queue[0] == event:
                heapq.heappop(self._queue)
            else:
                return

        # Ideal time never runs backward. Events are popped in t-order, so normally event.t >= _ideal_time
        # and this is just `= event.t`; the max() guards the degenerate case of an event scheduled in the
        # past (a t that has already elapsed), which fires immediately but must not drag the scheduler's
        # clock backward — that would throw off the timing of everything reading _ideal_time.
        self._ideal_time = max(self._ideal_time, event.t)
        # Record the wake time *now*, before running the action, so it marks when this event actually fired.
        # Relative timing measures the next wait from here (see _compute_wait_duration), so event spacing
        # tracks the requested durations regardless of how long each action's callback runs.
        # Skipped while timing is unanchored (pre-anchor events are immediate and off the schedule)
        if self._start_time is not None:
            self._last_event_time = self._time.now()
        logger.debug("Executing event %r scheduled at %s", event.metadata, event.t)
        try:
            event.action()
        except Exception as e:
            logger.exception(f"Error executing event '{event.metadata}': {e}")
