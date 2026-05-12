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

**One central `Scheduler` thread** running a heap of `(scheduler_time, action)` events. Each clock thread, on `wait()`, computes its absolute scheduler-time via `clock_to_scheduler_time` (walking up parents converting beats↔time↔parent_beat↔parent_time…), schedules a wakeup, and blocks. The scheduler pops the next event, sleeps until its time, fires `_wake_and_advance_to_next_wait`, then blocks on a condition until the woken clock has hit its next `wait` (preserving the original's single-clock-at-a-time property without nested wait loops).

This collapses:
- N nested wait loops → 1 scheduler loop
- per-clock `_queue` + `_queue_lock` → 1 heap
- `rouse_and_hold` ceremony → "modify the scheduled event in place" (or scheduler `hold`/`release`)
- "catch up all relatives" → just compute `beat()` lazily from scheduler time when asked
- ~8 pieces of per-clock state → ~2 (`_wait_event`, `_entering_wait_condition`)

## What cb2 has today

- `cb2/scheduler.py` — central scheduler with heap, hold/release, timing-policy blend (~190 lines, mostly done)
- `cb2/clock.py` — `Clock` with family tree, tempo properties (delegating to `TempoHistory`), `wait()`, `fork()`, `clock_to_scheduler_time` / `scheduler_to_clock_time` (~410 lines)
- `cb2/tempo_envelope.py` — original `TempoEnvelope`/`TempoHistory` ported, with an `@lru_cache` decorator on `time_at_beat`/`beat_at_time`
- `cb2/metric_phase.py` — `MetricPhaseTarget` ported essentially unchanged
- `cb2/enums.py` — `DurationUnits`, `TempoUnits`
- `cb2/test/mock_time.py` — clever compression-factor `time.sleep`/`time.time`/`threading.Event` mocks for fast deterministic tests

## What cb2 is missing

The architectural shape is right, but the unfinished pieces are the hard ones:

- **`bring_up_to_date` is `# TODO: pass`.** Looping envelopes and function-defined tempos don't work because the scheduler pre-computes the wake time, but the tempo curve only extends lazily.
- **No tempo-change-from-foreign-thread mechanism.** An already-scheduled wakeup is now wrong after a tempo change; it must be re-inserted in the heap. (Original handles this via `rouse_and_hold`.)
- **No `kill` / `ClockKilledError` / `DeadClockError`.** `child._killed = True` is set in `fork` but never initialized or checked.
- **No `current_clock()` integration** with a top-level `wait()` helper.
- **No `TimeStamp`, no synchronization policy.** cb2's `beat()` from a sibling thread is stale — but the right fix is lazy beat-from-scheduler-time, not the original's eager catch-up.
- **No `rouse_and_hold` analogue** for external-thread mutations.
- **Scheduler serializes everything.** `_execute_event` blocks on `_entering_wait_condition` until the woken clock hits its next wait. Original has the same property (master sleeps, children serialize behind the parent), so semantics match — but it's worth confirming.
- **No SCAMP-facing surface compatibility.** scamp imports `Clock`, `TempoEnvelope`, `TempoHistory`, `wait`, `fork`, `current_clock`, `TimeStamp`, `MetricPhaseTarget`, fast-forwarding, etc.
- **`lru_cache` on tempo conversions** is incompatible with lazy extension — caches will return stale answers as soon as the curve is extended. Must come out.

## Implementation plan (in dependency order)

### Step 1 — Lazy tempo extension in `TempoHistory`

Make `time_at_beat`, `beat_at_time`, and `advance` auto-extend looping envelopes / function-defined tempos on demand. Drop the `@lru_cache` decorator from cb2's `tempo_envelope.py` — caching and lazy extension can't coexist.

The `_envelope_loop_or_function` state currently lives on the original `Clock`; in cb2 it should move to `TempoHistory` so that any caller asking "what time is beat 5000?" on a looping clock gets the right answer without the clock having to be involved. `TempoHistory.extend_to(beat)` becomes the single extension primitive, called from inside the conversion methods. This is the keystone — Steps 2, 3, and 7 all assume it works.

Keep looping support for **both** `set_*_targets(loop=True)` (easy: append a copy of segments) **and** `apply_*_function(domain_end=None)` (the hard one: needs the existing extrema-and-inflection logic from `clockblocks/clock.py:1238-1287` ported into `TempoHistory`).

### Step 2 — Lazy `beat()` / `time()` from any thread

Instead of the original's eager "catch up all relatives on every wake", make `clock.beat()` and `clock.time()` compute on demand:

```
clock.time()  = scheduler_to_clock_time(scheduler.time(), units="time")
clock.beat()  = scheduler_to_clock_time(scheduler.time(), units="beats")
```

This already mostly works in cb2 — `scheduler_to_clock_time` is implemented. What's missing is that `tempo_history.beat()` / `tempo_history.time()` currently track only what the *owning thread* has called `advance()` for. Either:
- (a) drop those entirely and always go through the scheduler, or
- (b) keep them as the "committed" position (where the clock has actually run code up to), and add `beat_now()` / `time_now()` for the scheduler-derived live position.

Option (a) is cleaner. Synchronization policy disappears entirely.

### Step 3 — Reschedule-on-tempo-change

When a tempo setter runs against a clock whose wakeup is already queued (because it's dormant, or because the change affects a descendant's queued wakeup), the heap entry is now wrong. The fix:

1. Decorator `@tempo_modification` finds all heap entries whose `acting_clock` is `self` or a descendant of `self`.
2. Recomputes their `t` via `clock_to_scheduler_time` against the new tempo curve.
3. Replaces them in the heap (`heapq` doesn't support update-in-place — either re-heapify or use the standard "lazy deletion via sentinel" pattern).

This replaces `rouse_and_hold` for the common case (tempo changed from a non-clock thread). It also means the scheduler needs an API to atomically rewrite events for a given clock.

For tempo changes coming from a *sibling clock's* thread, same mechanism — the affected sibling is dormant and queued, just rewrite.

### Step 4 — Kill / DeadClockError / ClockKilledError

- Initialize `self._killed = False` in `__init__`.
- `kill()` sets `_killed`, removes any queued wakeups for self/descendants from the scheduler, sets `_wait_event` so the thread exits with `ClockKilledError`.
- `wait()` and `fork()` raise `DeadClockError` if `_killed`.
- Wrap the user function in `fork`'s `_process` with try/except for both errors (port from `clockblocks/clock.py:899-902`).
- Cascade kill to children.

### Step 5 — `current_clock()`, top-level `wait()`/`fork()`, `run_as_server`, `wait_forever`

Port from `clockblocks/utilities.py` and `clockblocks/clock.py`. cb2 already uses `threading.current_thread().__clock__ = child` in fork; just need the lookup helper and the module-level wrappers.

`run_as_server` is needed for interactive REPL usage (it backgrounds the master clock thread so the main thread stays interactive). Port verbatim.

### Step 6 — Fast-forward

Scheduler-side toggle. In `Scheduler.run`, when fast-forward is active, skip the wall-clock `sleep` and instead advance `_ideal_time` straight to the next event's `t`. Goal time can be set to `float('inf')` (full fast-forward) or to a specific scheduler-time. No more cheating with `_start_time`.

API on `Clock` mirrors original: `fast_forward()`, `fast_forward_to_time`, `fast_forward_in_time`, `fast_forward_to_beat`, `fast_forward_in_beats`, `is_fast_forwarding`.

### Step 7 — `TimeStamp`

Store `scheduler_time` at construction. Resolve per-clock beats lazily via Step 2 (`scheduler_to_clock_time` against each clock's tempo history). The master `time_stamp_data` dict goes away entirely — caching is unnecessary if resolution is cheap, and resolution is cheap because tempo histories are append-only past the committed point.

### Step 8 — External-thread mutation lock

cb2's `Scheduler.hold()` / `release()` already exist. Wire them into the `@tempo_modification` decorator: if `current_clock() != self` and `self` not in `current_clock().iterate_inheritance()`, briefly `scheduler.hold()` around the mutation. Step 3's reschedule logic handles the actual heap update; `hold()` just prevents the scheduler from firing a now-stale event while the heap is being rewritten.

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
- Fast-forward (Step 6)
- TimeStamp consistency across clocks (Step 7)

## Known design questions to revisit during implementation

- **Scheduler-action serialization.** Original semantics serialize per master clock; cb2 inherits this property because `_wake_and_advance_to_next_wait` blocks on `_entering_wait_condition`. Confirm this is what we want (the alternative — letting the scheduler issue another wakeup while a clock is still in user code — would allow real parallelism between clocks but break determinism users rely on).
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
