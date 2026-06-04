# cb2 → clockblocks 1.0: Redesign Plan

`cb2/` is a scratch space for a complete redesign of clockblocks that will eventually become clockblocks 1.0. This document captures why the redesign is happening, what the new architecture looks like, what's already in cb2/, and what's left to do.

## Why redesign

The original `clockblocks/` is correct but architecturally tangled. The core insanity:

1. **No central scheduler.** Each `Clock` owns a `_queue` of child `_WakeUpCall`s. Only the master calls `sleep_precisely_until`. When a child waits, it registers a wakeup time with its parent and `_wait_event.wait()`s. The parent's own `wait()` loop pops the child's wakeup, sleeps the master, signals the child, then waits for the child to hit its next wait before continuing. An N-deep clock tree means N threads parked in nested wait loops at any instant.

2. **Three different busy-wait sync points** (`_wait_for_children_to_finish_processing`, `_wait_for_child_to_finish_processing`, and the post-fork dormancy poll) spinning at `time.sleep(0.000001)`.

3. **`rouse_and_hold` + `_WaitKeeper`.** Foreign-thread or parent-thread mutations must wake the dormant clock early (`WokenEarlyError`/`_woken_early`), propagate the hold up the parent chain, do the mutation, then release. Every `@tempo_modification` setter wraps this dance.

4. **Eager "catch up all relatives" on every wake.** Each clock's `tempo_history` only advances when that clock wakes, so siblings/cousins drift. Every wakeup walks the family tree (`_catch_up_children`) advancing everyone's tempo curves, with timing warnings when it's slow.

5. **Per-clock state mountain:** `_queue`, `_queue_lock`, `_dormant`, `_wait_event`, `_woken_early`, `_wait_keeper`, `_envelope_loop_or_function`, `_fast_forward_goal`, `_priority_counter` — all coupled.

6. **Fast-forward** cheats by retroactively rewinding `_start_time`.

## Architectural pivot (already in cb2)

**One central `Scheduler` thread** running a heap of `(scheduler_time, action)` events. Each clock thread, on `wait()`, computes its absolute scheduler-time via `clock_to_scheduler_time` (walking up parents converting beats↔time↔parent_beat↔parent_time…), schedules a wakeup, and blocks. The scheduler pops the next event, sleeps until its time, fires `_wake_and_advance_to_next_wait_call`, then blocks on a condition until the woken clock has hit its next `wait` (preserving the original's single-clock-at-a-time property without nested wait loops).

This collapses:
- N nested wait loops → 1 scheduler loop
- per-clock `_queue` + `_queue_lock` → 1 heap
- `rouse_and_hold` ceremony → "modify the scheduled event in place" (or scheduler `hold`/`release`)
- "catch up all relatives" → just compute `beat()` lazily from scheduler time when asked
- ~8 pieces of per-clock state → ~2 (`_wait_event`, `_scheduler_park_condition`)

## What cb2 has today

- `cb2/scheduler.py` — central scheduler with heap, hold/release, timing-policy blend (~190 lines, mostly done)
- `cb2/clock.py` — `Clock` with family tree, tempo properties (delegating to `TempoHistory`), `wait()`, `fork()`, `clock_to_scheduler_time` / `scheduler_to_clock_time` (~410 lines)
- `cb2/tempo_envelope.py` — original `TempoEnvelope`/`TempoHistory` ported, with an `@lru_cache` decorator on `time_at_beat`/`beat_at_time`
- `cb2/metric_phase.py` — `MetricPhaseTarget` ported essentially unchanged
- `cb2/enums.py` — `DurationUnits`, `TempoUnits`
- `cb2/test/mock_time.py` — clever compression-factor `time.sleep`/`time.time`/`threading.Event` mocks for fast deterministic tests

## What cb2 is missing

The architectural shape is right, but the unfinished pieces are the hard ones:

- **No `kill` / `ClockKilledError` / `DeadClockError`.** `child._killed = True` is set in `fork` but never initialized or checked.
- **No `TimeStamp`, no synchronization policy.** cb2's `beat()` from a sibling thread is stale — but the right fix is lazy beat-from-scheduler-time, not the original's eager catch-up.
- **Scheduler serializes everything.** `_execute_event` blocks on `_scheduler_park_condition` until the woken clock hits its next wait. Original has the same property (master sleeps, children serialize behind the parent), so semantics match — but it's worth confirming.
- **No SCAMP-facing surface compatibility.** scamp imports `Clock`, `TempoEnvelope`, `TempoHistory`, `wait`, `fork`, `current_clock`, `TimeStamp`, `MetricPhaseTarget`, fast-forwarding, etc.

## Implementation plan (in dependency order)

### Step 1 — Lazy tempo extension in `TempoHistory`

**Status:** done. `extend_to` and `_extend_function_or_envelope_loop` are called from `time_at_beat` / `beat_at_time`; `bring_up_to_date` is implemented in `clock.py`.

**On `lru_cache`:** kept. The original PLAN said to drop it, claiming caching was incompatible with lazy extension. That was wrong — lazy `extend_to(b)` only adds segments past beat `b`, so a cached `time_at_beat(b)` stays valid. The cache is correctly invalidated on mutation by the `@tempo_modification` decorator wrapping every curve-mutating method (and the standalone `cache_clear()` pair in the manual `_beat`/`_t` setter). Hits save real work — `time_at_beat` does interval integration and `beat_at_time` does iterative root-finding to `max_error=1e-12`. Whether the workload hits the 32-entry cache often is empirical; revisit during profiling once Step 2 (lazy `beat()`/`time()`) and Step 7 (`TimeStamp`) land, since both produce access patterns where many callers query the same `(clock, scheduler_time)` simultaneously.

Make `time_at_beat`, `beat_at_time`, and `advance` auto-extend looping envelopes / function-defined tempos on demand.

The `_envelope_loop_or_function` state currently lives on the original `Clock`; in cb2 it should move to `TempoHistory` so that any caller asking "what time is beat 5000?" on a looping clock gets the right answer without the clock having to be involved. `TempoHistory.extend_to(beat)` becomes the single extension primitive, called from inside the conversion methods. This is the keystone — Steps 2, 3, and 7 all assume it works.

Keep looping support for **both** `set_*_targets(loop=True)` (easy: append a copy of segments) **and** `apply_*_function(domain_end=None)` (the hard one: needs the existing extrema-and-inflection logic from `clockblocks/clock.py:1238-1287` ported into `TempoHistory`).

### Step 2 — Lazy `beat()` / `time()` from any thread

**Status:** done. `Clock.beat()` and `Clock.time()` go through `scheduler_to_clock_time(scheduler.time(), ...)` so any thread gets a live position. `TempoHistory.beat()` / `.time()` are kept as the "committed pointer" — the position the owning thread's `wait()` has actually advanced to. This is the chosen split (option (b) of the original two options): clock-level reads are live, tempo-history-level reads are committed.

Why not Option (a) (drop committed pointer entirely): `wait()` and the tempo curve mutation logic in `tempo_history` (truncate / advance / append) all need a notion of "where in the curve are we right now from the perspective of this curve's append-only history". That's the committed pointer. Removing it would require restructuring `TempoHistory` to be stateless about position, which is a bigger surgery than is warranted.

Implementation notes:
- `wait()` now uses `self.tempo_history.beat()` (committed) on the post-wait advance line, since `self.beat()` is live and would make the delta zero.
- `current_time_in_scheduler()` was removed. With lazy reads it became a round-trip back to `self.scheduler.time()`, and putting it on `Clock` misleadingly suggests different clocks could have different scheduler times. Use `clock.scheduler.time()` at callsites instead.
- `bring_up_to_date()` already does the right thing: `delta = scheduler_to_clock_time(...) - tempo_history.beat()`.

Reads do not mutate `tempo_history` — that would be a write through a getter and would race between threads. The owning thread's `wait()` is the only path that advances the committed pointer.

### Step 3 — Reschedule-on-tempo-change

**Status:** done. `Scheduler.reschedule(matches, recompute)` walks the heap and re-heapifies; `Clock._reschedule_self_and_descendants()` matches by `acting_clock` and recomputes via `clock_to_scheduler_time`; `@_reschedule_after_tempo_change` decorator wraps the tempo/rate/beat_length setters and brackets the mutation with the locks described in Step 8 (originally `scheduler.held()`, now `while_quiescent()` / `_tree_lock`).

When a tempo setter runs against a clock whose wakeup is already queued (because it's dormant, or because the change affects a descendant's queued wakeup), the heap entry is now wrong. The fix:

1. Decorator `@tempo_modification` finds all heap entries whose `acting_clock` is `self` or a descendant of `self`.
2. Recomputes their `t` via `clock_to_scheduler_time` against the new tempo curve.
3. Replaces them in the heap (`heapq` doesn't support update-in-place — either re-heapify or use the standard "lazy deletion via sentinel" pattern).

This replaces `rouse_and_hold` for the common case (tempo changed from a non-clock thread). It also means the scheduler needs an API to atomically rewrite events for a given clock.

For tempo changes coming from a *sibling clock's* thread, same mechanism — the affected sibling is dormant and queued, just rewrite.

### Step 4 — Kill / DeadClockError / ClockKilledError

**Status:** done. Three-state `ClockState` enum (PENDING/ALIVE/DEAD) replaces the original `_killed` flag plan. `ClockblocksError` base, `ClockKilledError`, `DeadClockError`, and `WrongThreadError` are defined; `kill()` cascades, removes pending wakeups *and* pending forks via new `Scheduler.remove_events`, wakes any parked wait, releases the scheduler, and eagerly detaches from the parent's `_children`. `wait()` rejects non-ALIVE clocks and cross-thread calls; `fork()` rejects non-ALIVE. `_process` catches both lifecycle errors. Tests: `tests/test_kill.py` (11) + `tests/test_fork.py` (4).

### Step 4.5 — Legibility pass on wait() / fork() / kill()

**Status:** done. Applied the `wait()` STEP shape to `fork()` and extracted one helper; deliberately left `wait()` and `kill()` as-is.

- `fork()`: extracted `_resolve_start_delay(schedule_at)` (clock.py) — the numeric-beat / `MetricPhaseTarget` / `None` branching is now a self-contained method, so fork()'s body reads as create-child → resolve-delay → define-lifecycle → schedule.
- `fork()`: gave the body STEP 1–4 comments and the inner `_process` closure Step 3a–3e sub-comments (setup / run user fn / cleanup / release scheduler / done_callback). Also collapsed the two identical `except ClockKilledError / except DeadClockError` blocks into one `except (ClockKilledError, DeadClockError)`.
- `wait()`: left unchanged. It's already the model the others are measured against; extracting STEP 2/4 would mean threading `wake_up_beat` through a helper return value for marginal gain.
- `kill()`: left unchanged. Its STEP 1–3 comments already match the `wait()`/`fork()` shape.

Decision: stopped here rather than factor further — the closures capture too much (`child`, `self`, `process_function`, `args`, `done_callback`, `start_delay`) to extract cleanly into methods, and STEP comments deliver the readability without the parameter-passing ceremony.

### Step 5 — `current_clock()`, top-level `wait()`/`fork()`, `run_as_server`, `wait_forever`

**Status:** done. `Clock.fork_unsynchronized` / `wait_forever` / `wait_for_children_to_finish` /
`run_as_server` (clock.py), plus module-level `fork` / `fork_unsynchronized` / `wait_forever` /
`wait_for_children_to_finish` wrappers in `cb2/utilities.py` delegating to `current_clock()`
(`current_clock()` and module `wait()` already existed). Tests: `tests/test_module_api.py` (8).

Not a verbatim port — the originals leaned on the abandoned per-parent-queue / `rouse_and_hold` /
`_wait_keeper` machinery. Adapted to the scheduler model:
- `wait_forever` loops `self.wait(1.0)`, breaking on `ClockKilledError` / `DeadClockError`.
- `wait_for_children_to_finish` loops `self.wait(1.0)` until `self._children` is empty. The caller
  must keep yielding to the scheduler (via `wait`) so the children can actually run, so this polls
  at 1-beat granularity rather than waking exactly when the last child ends. Good enough for a
  caller that has nothing left to do; a precise "wake me when children finish" signal (the old
  rouse mechanism) is deferred.
- `fork_unsynchronized` spawns a plain daemon `Thread` (no clock bound to it). Step 9's thread pool
  will replace the raw `Thread` with `_run_in_pool`.
- `run_as_server` backgrounds the master on a daemon thread that takes ownership
  (`__clock__ = self`) and calls `wait_forever`; the calling thread relinquishes ownership
  (`__clock__ = None`) and returns. Adapted (not verbatim) to cb2's `current_clock()`/`wait()`.

Still TODO from the original surface: `fork()`'s "pass the clock as first arg" convenience and
`fork_unsynchronized` pool routing (Step 9).

### Step 6 — Fast-forward

**Status:** done. Scheduler-side toggle, with all the Clock API mirroring the original. Tests:
`tests/test_fast_forward.py` (13).

Scheduler side (`scheduler.py`): a `_fast_forward_goal` (a *scheduler-time*, `float('inf')` for
indefinite, or `None`) set via `set_fast_forward_goal()` (which notifies the run loop so a goal change
takes effect immediately rather than after the current timed wait). In the run loop, STEP 1b decides each
event with two helpers:
- `_fast_forwarding_through(next_event)` — pure predicate: is a goal set and `next_event.t < goal`? If so
  the wait is skipped (`_ideal_time` leaps to the event when it fires) and `_was_fast_forwarding` is set.
- otherwise `_end_fast_forward_if_active(now)` settles any fast-forward that was in progress before the
  wait is timed normally. It's a no-op unless FF is ending one of two ways: (1) a finite goal reached
  (`next_event.t >= goal`) → advance `_ideal_time` to the goal, clear it, re-anchor, then time the
  remaining `goal -> event` span normally (zero at the boundary, so an event landing exactly on the goal
  still fires instantly — matching the original, where *reaching* the goal ends FF); or (2) FF switched
  off externally (goal cleared mid-flight), detected via `_was_fast_forwarding` → re-anchor. Both endings
  clear `_was_fast_forwarding`, so an early wakeup during the post-goal real-time tail can't re-anchor a
  second time and drop the already-elapsed wait.

`_reanchor_timing(now)` re-pegs `_last_wake_time = now` and `_start_time = now - _ideal_time` so both the
relative and absolute timing policies resume cleanly. This replaces the original's `_start_time`-rewinding
cheat: cb2's `time()`/`beat()` derive from `_ideal_time` (not wall clock), so nothing needs faking to keep
the clock position correct — re-anchoring only restores the *timing-policy* reference points.

API on `Clock` (master-only, raising `NotMasterClockError` off-master) mirrors original: `fast_forward()`,
`fast_forward_to_time`, `fast_forward_in_time`, `fast_forward_to_beat`, `fast_forward_in_beats`,
`is_fast_forwarding`. The `*_to_*` methods convert the requested clock time/beat to scheduler-time via
`clock_to_scheduler_time` and reject targets in the past; `is_fast_forwarding` reflects the shared
scheduler state, so it's true for the whole family at once.

### Step 7 — `TimeStamp`

**Status:** done. `cb2/time_stamp.py` defines `TimeStamp` as a thin wrapper around a captured
`scheduler_time` plus the family's master. `beat_in_clock(c)` / `time_in_clock(c)` go through
`c.scheduler_to_clock_time(self.scheduler_time, desired_units=...)`; `time_in_master` is a
convenience for `time_in_clock(master)` (it is **not** equal to `scheduler_time` — the master
has a `parent_offset == scheduler.time()` at its construction). Foreign-family clocks are
rejected. Equality / ordering compare on `scheduler_time` alone — `wall_time` was dropped
(scamp never read it; under fast-forward it was a non-deterministic tiebreaker anyway). The
old master-side `time_stamp_data` dedup cache is gone — resolution is cheap. Tests:
`tests/test_time_stamp.py` (6).

**Kept distinct from `Moment`** (vs. subsuming it into an absolute Moment): different intent
(captured-past vs declared-future), different shape (clock-agnostic vs anchored to one
clock's beat/time axis), and scamp's transcriber use-case wants exactly the clock-agnostic
shape — store one scheduler-time, project into many clocks later.

### Step 8 — External-thread mutation lock

**Status:** done. The original `Scheduler.held()` approach (an unconditional coarse pause via a `threading.Event` gate) is **gone** — it was non-reentrant and stompable (two holders, e.g. a tempo change racing a kill, clobbered each other), and being level-checked at the top of the run loop it couldn't even preempt an action already in flight, so it didn't actually stop the scheduler from waking a clock and racing its `tempo_history` mid-rewrite.

Replaced by a three-primitive model (see the block comment at the top of `Scheduler.__init__`, and the `_tree_lock` property in `clock.py`):

- **`_queue_change_condition`** (scheduler) — guards the heap and signals changes to it. The run loop now holds it across peek+compute+wait as a single critical section, fixing a **lost-wakeup**: previously `_get_next_event` and `_wait_for` were two separate acquisitions, so a `reschedule` landing in the gap notified into the void and the loop slept the stale (longer) duration, firing the event late.
- **`_execution_lock`** (scheduler) — held by the run loop for the full duration of each action's execution, so "held" == "an action is running". Exposed via the `while_quiescent()` context manager: an external thread takes it to mutate action-related state only when no action is in flight. (It can be held across `_execute_event` — unlike the queue lock — precisely because a running clock never needs it, only external mutators do.)
- **`_clock_tree_lock`** (per family, on the master; via `Clock._tree_lock`) — serializes structural ops on the clock tree. `fork()` (create+append+schedule) and `kill()` (collect+flag+remove) are now atomic against each other, so a fork can't be orphaned by a concurrent kill; `_fork_wrapper` cleanup detaches under it too.

`_reschedule_after_tempo_change` branches on `current_clock()`: an **external** thread (`current_clock() is None`) takes `while_quiescent()` then `_tree_lock`; a call **on a clock's own thread** takes only `_tree_lock`, since the scheduler is already frozen on that clock (a clock-thread call to `while_quiescent()` would self-deadlock). Lock order is `_execution_lock` → `_tree_lock` → `_queue_change_condition`, and the order matters — see the decorator docstring for the deadlock it avoids.

This also hardened Step 3 (the reschedule now runs under `_tree_lock`, so descendant enumeration can't race a concurrent fork/kill) and Step 4 (`kill()` no longer uses the scheduler hold; it relies on `_tree_lock` plus the invariant that every victim observes `DEAD`).

### Step 9 — Thread-pool for forks

Port `_run_in_pool` + `_pool_semaphore` from original (`clockblocks/clock.py:793-801`). Real perf win when many short forks happen (which scamp does constantly for note playback). Threads-per-fork (cb2's current approach) is fine for correctness but slow.

**Longer-term, eliminate `fork_unsynchronized` for the common case.** Its main use in scamp is high-frequency, low-overhead bursts — glissandi and continuous expression/volume curves — i.e. many MIDI messages with tiny waits between them. The new `Clock.schedule_action()` (leaf callbacks fired directly on the scheduler — no child clock, no per-step scheduler/clock context-switch handoff) is the intended replacement for that pattern: sample the envelope and schedule N sends as leaf events instead of forking a thread that micro-waits. The thread pool here is a stopgap for genuinely thread-y work; the gliss/expression path should move to `schedule_action`. Benchmark first — the single-sleeper scheduler may already remove most of the old per-clock busy-wait lag that motivated `fork_unsynchronized` in the first place.

### Step 10 — SCAMP integration surface

scamp imports (verify against `scamp/src/scamp/__init__.py`):
- `Clock`, `TempoEnvelope`, `TempoHistory`, `MetricPhaseTarget`
- `wait`, `wait_for_children_to_finish`, `wait_forever`, `fork`, `fork_unsynchronized`, `current_clock`
- `TimeStamp`
- Plus all the `set_tempo_target` / `apply_*_function` family on `Clock`

Run `scamp/test/test_examples.py` as the integration test before declaring done. Expect minor output diffs (timing precision); review and regenerate goldens.

### Step 10.5 — Bridge missing `Clock` tempo-target / tempo-function methods

**Status:** done. cb2's `TempoHistory` already had the underlying methods; this step added the
`Clock`-level bridges, gave the duration param a `ResolvableMoment`-first signature with a
back-compat deprecation path for bare numbers, and converted the obsoleted-by-design legacy APIs
(synchronization/timing policy, rouse_and_hold/release_from_suspension) to raise-on-access stubs
with explanatory messages. `Clock.time_in_master()` is a one-line proxy to `master.time()`.
`log_processing_time` / `stop_logging_processing_time` deferred.

Original scope (kept for the record):

- `Clock.set_beat_length_target` / `set_rate_target` / `set_tempo_target`
- `Clock.set_beat_length_targets` / `set_rate_targets` / `set_tempo_targets` (plural, loopable)
- `Clock.apply_beat_length_function` / `apply_rate_function` / `apply_tempo_function`
- `Clock.stop_tempo_loop_or_function`
- `Clock.time_in_master` — trivial convenience: `self.scheduler_to_clock_time(scheduler.time(), 'time')`
  projected to master, or just `master.time()` evaluated from this thread.

Each bridge needs the `@_reschedule_after_tempo_change` wrap (we already use it for the
beat_length/rate/tempo setters) so a queued descendant wakeup gets re-projected against the new curve.

**`duration` should be a `Moment` / list of `Moment`s now.** In the redesign, the `when`/`duration`
vocabulary unified around `Moment.after_beats` / `after_time` / `at_beat` / `at_time`. The set-target
methods predate that and take raw numbers + `duration_units="beats"|"time"` kwarg. New shape:

```python
clock.set_tempo_target(100, Moment.after_beats(9))    # equivalent to legacy (..., 9)
clock.set_tempo_target(100, Moment.after_time(2.5))   # what duration_units="time" used to mean
```

For backwards compatibility accept a bare number too, but emit `DeprecationWarning("pass
Moment.after_beats(n) instead of a bare number; duration_units is going away")`. After a release,
drop `duration_units` entirely. The plural forms (`set_*_targets`) become a list of `Moment`s.

Deferred (uncertain whether they translate to the central-scheduler model):
- `log_processing_time` / `stop_logging_processing_time`

Already-gone-by-design (do not port). Each should remain as an attribute/method on `Clock` that
**raises a clear error explaining what to do instead** when accessed, rather than silently being
absent — otherwise users get `AttributeError: 'Session' object has no attribute 'X'` and have to
guess. Suggest a small `_removed_attribute(name, replacement, reason)` helper that raises
`AttributeError(f"{name} was removed in clockblocks 1.0: {reason}. Use {replacement} instead.")`:

- `synchronization_policy` (prop+setter) — obsoleted by Step 2's lazy `beat()` (any thread already
  sees the live position; there's nothing to "synchronize"). No replacement; just gone.
- `timing_policy` (prop+setter) and `use_absolute_timing_policy` / `use_relative_timing_policy` /
  `use_mixed_timing_policy` — moved to `Scheduler` because timing is now a scheduler-wide property.
  Replacement: `scheduler.timing_policy = ...` (and a future `use_*_timing_policy()` on Scheduler
  if we want to mirror the legacy convenience methods).
- `rouse_and_hold` / `release_from_suspension` — replaced by `Clock.while_scheduler_quiescent()`
  (the `with`-block form covers both halves atomically and is exception-safe).

### Step 11 — Unit tests

Keep cb2's `mock_time.py` compression trick. Cover:
- Single-clock wait timing under each timing_policy
- Nested fork (child, grandchild) timing
- Tempo change from owning thread
- Tempo change from sibling thread (Step 3)
- Tempo change from non-clock thread (Step 8)
- Looping envelope wrap-around (Step 1)
- Function-defined tempo extension across a wait boundary (Step 1)
- Kill cascades (Step 4)
- Tempo change reschedules a pending `schedule_at` fork (Step 3)
- Fast-forward (Step 6)
- TimeStamp consistency across clocks (Step 7)

## Known design questions to revisit during implementation

- **Scheduler-action serialization.** Original semantics serialize per master clock; cb2 inherits this property because `_wake_and_advance_to_next_wait_call` blocks on `_scheduler_park_condition`. Confirm this is what we want (the alternative — letting the scheduler issue another wakeup while a clock is still in user code — would allow real parallelism between clocks but break determinism users rely on).
- **`schedule_at` with `MetricPhaseTarget`** at fork time needs the parent's `beat()` to be accurate at the moment of the fork call. Step 2 (lazy beat) handles this for the foreign-thread case; for the owning-thread case it's already correct.
- **Priority ordering.** Original uses `_priority_counter` to break ties between sibling clocks woken at the same beat (earlier-forked first). cb2's `priority: Tuple[int, ...]` in `QueueEvent` is in the right shape; need to make sure `clock_id` (which is constructed as the parent's clock_id plus a counter) gives the same ordering as original's priority.
- **Multiple master clocks.** Original `get_scheduler()` returns a module-level singleton. Two `Clock()` calls with `parent=None` would share a scheduler. Is that desired? Probably yes (one timing source per process), but worth a sanity check.
- **Natural-end master strands the shared scheduler (the "B" question).** When a master clock's top-level code just falls off the end *without* a `kill()`, there is no `_fork_wrapper` cleanup, so the scheduler thread is left parked on that master's `_scheduler_park_condition` forever (it was waiting for a next `wait()` that never comes). The `kill()` docstring waves this away — "since it's a daemon thread and the process is exiting, this doesn't matter" — but that assumption is **false the moment a second master is created in the same live process**: the singleton scheduler is still parked on the dead first master, so the second master's initial wake never fires and `Clock.__init__`'s `self._wait_event.wait()` deadlocks. Surfaced concretely by `scamp/test/test_examples.py`, which imports ~50 example scripts serially into one process: every example that didn't `kill()` its `Session` hung the *next* one. (Worked around harness-side for now — the test runner kills the active master between examples — but that only patches scamp's tests, not the underlying cb2 behavior.) This is the same gap flagged in the Step 4 notes ("MASTER, natural end … will matter once run_as_server lands — Step 5 will need its own cleanup wrapper"). Decide the intended lifecycle for a master whose owning thread finishes: candidate fixes include (a) a master-side cleanup hook analogous to `_fork_wrapper` (atexit and/or weakref finalizer that releases/kills the master), (b) `get_scheduler()` / new-master construction supersedes-and-kills any stale parked master, or (c) make the master a context manager so `with Clock()/Session():` guarantees teardown. (a) and (b) are the only ones that fix the silent-fall-off-the-end case without requiring user code changes.

## Out of scope for 1.0

- Parallelism between sibling clocks (see serialization note above).
- Sub-microsecond precision improvements.
- Network sync / distributed clocks.

## Done criteria

1. All Step-11 unit tests pass.
2. `scamp/test/test_examples.py` passes with goldens regenerated and diffs reviewed.
3. `cb2/` renamed to `clockblocks/` (after atomic swap with the existing dir, which moves to `clockblocks_legacy/` or a tag for one release).
4. Old `_WaitKeeper`, `rouse_and_hold`, `_woken_early`, `_synchronization_policy`, and the three busy-wait spin loops are gone from the new codebase.

## Possible 1.5 features

Features that aren't required for 1.0 parity but feel natural to add given the redesign,
and that the new architecture should make easier than the old one did.

### `ScheduledMoment` and `schedule_at` by time — **landed early as `Moment`**

This was originally a 1.5 idea but ended up implemented as part of the wait/fork-`when` work, not
deferred. Both extensions are done (`cb2/moment.py`):

- The "when" vocabulary is a **`ResolvableMoment`**: `Moment.at_beat` / `at_time` / `after_beats` /
  `after_time`, or a `MetricPhaseTarget` (which now also `resolve()`s, in beats or time). Scheduling by
  time vs beat is just a `Moment`'s `units`. A bare number is accepted only where the method name fixes
  its meaning: `wait(duration)` (relative) and `wait_until(when)` (absolute). `fork(when=…)` and
  `schedule_action(when=…)` *require* an explicit Moment — a bare number there is rejected with a
  `TypeError`, since "when" alone wouldn't say whether it's relative or absolute.
- Event metadata stores the absolute `target_moment`, and `_reschedule_self_and_descendants` calls
  `moment.scheduler_time(acting_clock)` on tempo changes — preserving beat *or* time as the moment
  dictates. New kinds of moment can be added without touching the reschedule logic.

**Resolved (Step 7).** `TimeStamp` is kept as a distinct primitive. An absolute `Moment` is
anchored to one clock's beat/time axis (forward-facing — "schedule at this point on this
clock"); a `TimeStamp` stores only a scheduler-time and projects into any clock in the family
(backward-facing — "this happened, what beat/time was it on each clock?"). Scamp's
transcriber wants the latter shape.

### Externally-driven scheduler clock

Let the scheduler's notion of "now" be driven by an external thread rather than wall-clock sleep.
Use case: drive the clock tree from Logic (or any DAW / external transport) so that scamp's
playback follows the host instead of free-running.

This should be a scheduler-side change only: replace (or augment) the `sleep_until(t)` step in
`Scheduler.run` with "wait until the external source signals that scheduler-time has reached `t`."
Concretely: an injectable time source (default = monotonic wall clock; alternative = a callable
that blocks until the host has advanced to a given beat/time). Clock-side code is unaffected.

Open questions: how does the external source map its own time to scheduler time on startup
(offset + rate)? What happens on host tempo changes — does the scheduler see them as a continuous
mapping, or are we treating the host as a discrete tick stream? Worth prototyping against MIDI
clock / MTC before committing to a shape.
