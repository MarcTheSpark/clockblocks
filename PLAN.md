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

**Status:** partially done. `current_clock()` and module-level `wait()` exist in `cb2/utilities.py`. Still TODO: top-level `fork()` / `fork_unsynchronized()`, `wait_forever`, `wait_for_children_to_finish`, `run_as_server`.

Port from `clockblocks/utilities.py` and `clockblocks/clock.py`. cb2 already uses `threading.current_thread().__clock__ = child` in fork; just need the lookup helper and the module-level wrappers.

`run_as_server` is needed for interactive REPL usage (it backgrounds the master clock thread so the main thread stays interactive). Port verbatim.

### Step 6 — Fast-forward

Scheduler-side toggle. In `Scheduler.run`, when fast-forward is active, skip the wall-clock `sleep` and instead advance `_ideal_time` straight to the next event's `t`. Goal time can be set to `float('inf')` (full fast-forward) or to a specific scheduler-time. No more cheating with `_start_time`.

API on `Clock` mirrors original: `fast_forward()`, `fast_forward_to_time`, `fast_forward_in_time`, `fast_forward_to_beat`, `fast_forward_in_beats`, `is_fast_forwarding`.

### Step 7 — `TimeStamp`

Store `scheduler_time` at construction. Resolve per-clock beats lazily via Step 2 (`scheduler_to_clock_time` against each clock's tempo history). The master `time_stamp_data` dict goes away entirely — caching is unnecessary if resolution is cheap, and resolution is cheap because tempo histories are append-only past the committed point.

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

### Step 10 — SCAMP integration surface

scamp imports (verify against `scamp/src/scamp/__init__.py`):
- `Clock`, `TempoEnvelope`, `TempoHistory`, `MetricPhaseTarget`
- `wait`, `wait_for_children_to_finish`, `wait_forever`, `fork`, `fork_unsynchronized`, `current_clock`
- `TimeStamp`
- Plus all the `set_tempo_target` / `apply_*_function` family on `Clock`

Run `scamp/test/test_examples.py` as the integration test before declaring done. Expect minor output diffs (timing precision); review and regenerate goldens.

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

### `ScheduledMoment` and `schedule_at` by time

The `schedule_at` parameter on `fork()` currently accepts a beat number or a `MetricPhaseTarget`.
Two extensions worth doing:

1. **Allow scheduling by time instead of beat.** Any clock with `tempo != 60` has diverging
   beat and time axes; "fire at beat N" and "fire after N seconds" are different points.
   Currently `schedule_at` only speaks beats. API options: a `units="time"|"beats"` kwarg
   alongside `schedule_at`, or a thin wrapper type.

2. **Introduce a `ScheduledMoment` (working name) class** that unifies the various ways to
   specify a future point in time:
   - fixed beat
   - fixed time
   - `MetricPhaseTarget` (next occurrence of a phase within a cycle)
   - possibly other future kinds (e.g. "at the next downbeat after beat X")

   This is conceptually distinct from `TimeStamp` (Step 7): a `TimeStamp` is a *resolved* moment
   that's already pinned to scheduler time and can be translated to any clock's frame.
   A `ScheduledMoment` is a *proposal* — a recipe for computing when something should happen,
   which may need re-evaluation if tempo changes between when it's specified and when it fires.

   The fork-event metadata would store the `ScheduledMoment` rather than a raw `target_beat`,
   and `_reschedule_self_and_descendants` would call `moment.resolve(parent_clock)` to recompute
   scheduler time on tempo changes. Same hook would let us add new kinds of moments later without
   touching the reschedule logic.

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
