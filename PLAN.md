# cb2 → clockblocks 1.0: Redesign Plan

`cb2/` is a scratch space for a complete redesign of clockblocks that will eventually become clockblocks 1.0. This document captures why the redesign is happening, the new architecture, and what's left to do.

## Why redesign

The original `clockblocks/` is correct but architecturally tangled:

1. **No central scheduler.** Each `Clock` owns a `_queue` of child wakeups; only the master sleeps. A child registers a wakeup with its parent and blocks; the parent pops it, sleeps, signals the child, waits for the child to reach its next wait, then continues. An N-deep tree means N threads parked in nested wait loops.
2. **Three busy-wait sync points** spinning at `time.sleep(0.000001)`.
3. **`rouse_and_hold` + `_WaitKeeper`.** Foreign/parent-thread mutations must wake the dormant clock early, propagate a hold up the chain, mutate, then release. Every `@tempo_modification` setter wraps this dance.
4. **Eager "catch up all relatives" on every wake.** A clock's `tempo_history` only advances when it wakes, so siblings drift; every wakeup walks the family tree advancing everyone.
5. **Per-clock state mountain:** `_queue`, `_queue_lock`, `_dormant`, `_wait_event`, `_woken_early`, `_wait_keeper`, `_envelope_loop_or_function`, `_fast_forward_goal`, `_priority_counter` — all coupled.
6. **Fast-forward** cheats by retroactively rewinding `_start_time`.

## The architecture (already in cb2)

**One central `Scheduler` thread** running a heap of `(scheduler_time, action)` events. On `wait()`, a clock computes its absolute scheduler-time via `clock_to_scheduler_time` (walking up parents), schedules a wakeup, and blocks. The scheduler pops the next event, sleeps until its time, wakes the clock, then blocks on a condition until that clock reaches its next `wait` (preserving the original's single-clock-at-a-time property without nested wait loops).

This collapses: N nested wait loops → 1 scheduler loop; per-clock `_queue` → 1 heap; `rouse_and_hold` → "modify the scheduled event in place"; "catch up all relatives" → compute `beat()` lazily from scheduler time when asked; ~8 pieces of per-clock state → ~2.

### Files in cb2

- `scheduler.py` — central scheduler: heap, hold/release, timing-policy blend, fast-forward
- `clock.py` — `Clock`: family tree, tempo properties (delegating to `TempoHistory`), `wait()`, `fork()`, `kill()`, `clock_to_scheduler_time` / `scheduler_to_clock_time`, lifecycle context manager
- `tempo_envelope.py` — `TempoEnvelope`/`TempoHistory`, lazy extension, `@lru_cache` on conversions
- `time_stamp.py` — `TimeStamp` (captured scheduler-time, projects into any clock in the family)
- `moment.py` — `ResolvableMoment` (`at_beat`/`at_time`/`after_beats`/`after_time`), the unified "when" vocabulary
- `metric_phase.py` — `MetricPhaseTarget`
- `enums.py` — `DurationUnits`, `TempoUnits`, `ClockState`
- `utilities.py` — module-level `wait`/`fork`/`fork_unsynchronized`/`wait_forever`/`wait_for_children_to_finish`/`current_clock`
- `test/mock_time.py` — compression-factor `time` / `Event` mocks for fast deterministic tests

---

## Completed steps

Tight summaries of the key decision in each; see git history and code comments for full detail.

- **Step 1 — Lazy tempo extension.** `extend_to` / `_extend_function_or_envelope_loop` called from `time_at_beat`/`beat_at_time`; `_envelope_loop_or_function` moved to `TempoHistory`. Supports both `set_*_targets(loop=True)` and `apply_*_function(domain_end=None)` (ported extrema/inflection logic). `lru_cache` **kept** — lazy `extend_to(b)` only appends past beat `b`, so cached conversions stay valid; invalidated on mutation by `@tempo_modification`.

- **Step 2 — Lazy `beat()` / `time()` from any thread.** Clock-level reads (`Clock.beat()`/`.time()`) go through `scheduler_to_clock_time(scheduler.time())` → live position from any thread. `TempoHistory.beat()`/`.time()` are the **committed pointer** — the position the owning thread's `wait()` has advanced to (kept because `wait()` and curve mutation need it). Reads never mutate `tempo_history`. `current_time_in_scheduler()` removed — use `clock.scheduler.time()`.

- **Step 3 — Reschedule-on-tempo-change.** `Scheduler.reschedule(matches, recompute)` re-heapifies; `Clock._reschedule_self_and_descendants()` matches by `acting_clock` and recomputes via `clock_to_scheduler_time`. The `@_reschedule_after_tempo_change` decorator brackets tempo/rate/beat_length setters with the Step-8 locks. Replaces `rouse_and_hold` for tempo changes from non-owning threads.

- **Step 4 — Kill / lifecycle errors.** Three-state `ClockState` enum (PENDING/ALIVE/DEAD). `ClockblocksError` base + `ClockKilledError`, `DeadClockError`, `WrongThreadError`. `kill()` cascades, removes pending wakeups *and* forks (`Scheduler.remove_events`), wakes parked waits, detaches from parent. `wait()`/`fork()` reject non-ALIVE and cross-thread calls. Tests: `test_kill.py`, `test_fork.py`.

- **Step 4.5 — Legibility pass.** Applied the `wait()` STEP shape to `fork()`, extracted `_resolve_start_delay`. Left `wait()`/`kill()` as-is (already the model).

- **Step 5 — Module API + server.** `current_clock()`, module-level `wait`/`fork`/`fork_unsynchronized`/`wait_forever`/`wait_for_children_to_finish` (delegating to `current_clock()`), `run_as_server`. Adapted to the scheduler model (not verbatim ports). `wait_for_children_to_finish` polls at 1-beat granularity; a precise "wake when children finish" signal is deferred. Tests: `test_module_api.py`.

- **Step 6 — Fast-forward.** Scheduler-side `_fast_forward_goal` (a scheduler-time) set via `set_fast_forward_goal()`. Run loop skips timed waits while fast-forwarding and settles cleanly when a finite goal is reached or the goal is cleared mid-flight (`_reanchor_timing` re-pegs the timing-policy reference points). Replaces the original's `_start_time`-rewinding cheat — cb2's `time()`/`beat()` derive from `_ideal_time`, so nothing needs faking. Master-only API mirrors original (`fast_forward*`, `is_fast_forwarding`). Tests: `test_fast_forward.py`.

- **Step 7 — `TimeStamp`.** Thin wrapper around a captured `scheduler_time` + the family master. `beat_in_clock(c)`/`time_in_clock(c)` go through `c.scheduler_to_clock_time(...)`. Equality/ordering on `scheduler_time` alone (`wall_time` dropped). Kept **distinct from `Moment`**: captured-past + clock-agnostic vs declared-future + anchored to one clock — scamp's transcriber wants the former. Tests: `test_time_stamp.py`.

- **Step 8 — External-thread mutation lock.** Replaced the original's stompable `held()` gate with three primitives: `_queue_change_condition` (guards heap, held across peek+compute+wait to fix a lost-wakeup), `_execution_lock` (held while an action runs; exposed as `while_quiescent()` for external mutators), `_clock_tree_lock` (per-family, serializes `fork`/`kill` structural ops). `_reschedule_after_tempo_change` branches on `current_clock()`: external thread takes `while_quiescent()` + `_tree_lock`; own-thread call takes only `_tree_lock` (else self-deadlock). Lock order: `_execution_lock` → `_tree_lock` → `_queue_change_condition`. Hardened Steps 3 and 4.

- **Step 10.5 — Clock tempo-target / tempo-function bridges.** Added `Clock`-level bridges over `TempoHistory`'s methods (`set_*_target(s)`, `apply_*_function`, `stop_tempo_loop_or_function`), each `@_reschedule_after_tempo_change`-wrapped. `duration` param takes a `ResolvableMoment` (`Moment.after_beats(n)`/`after_time(t)`) with a `DeprecationWarning` back-compat path for bare numbers (`duration_units` going away). `Clock.time_in_master()` proxies `master.time()`. Obsoleted legacy APIs raise explanatory errors instead of `AttributeError`: `synchronization_policy` (gone — Step 2 made it moot), `rouse_and_hold`/`release_from_suspension` (→ `while_scheduler_quiescent()`). `timing_policy` + `use_*_timing_policy` **kept as real methods** forwarding to `self.scheduler.timing_policy` (settable only on master). `log_processing_time` deferred.

- **Step 12 — Master lifecycle context manager.** `Clock.__enter__`/`__exit__` (`__exit__` kills, suppressing only `ClockKilledError`/`DeadClockError`); `with Session() as s:` for free. Terminal parks (`wait_forever`/`wait_for_children_to_finish`) now **propagate** `ClockKilledError` rather than swallowing it, so every wait variant behaves the same (kill mid-wait raises, caught at `_fork_wrapper` for sub-clocks or `__exit__`/user `try/except` for the master). Rationale: bare-script shutdown is Ctrl-C/process-exit, never `ClockKilledError`; only a *programmatic* `kill()` raises, exactly when an exception is the right signal. `run_as_server`'s loop absorbs the kill so its daemon thread exits quietly. Opt-in — main-thread scripts relying on thread-end teardown stay valid. scamp's `performance.py` call site reviewed, needs no change. Tests pass (86).

- **Documentation pass — largely done.** Docstrings made user-facing (implementation detail pushed to comments), wording ported from original clockblocks where still applicable, enums + `Scheduler` documented, clock-termination walkthrough moved to a module-level lifecycle comment, GPL-3.0 headers added.
  - **Deferred to the rename:** qualify bare cross-*module* xrefs (`:class:`Clock`` → `:class:`~clockblocks.clock.Clock``). The docs are one combined scamp+clockblocks Sphinx build with no intersphinx, so cross-module links work only when fully qualified. Pair with the `cb2`→`clockblocks` rename, since every `~cb2.…` prefix has to be rewritten to `~clockblocks.…` anyway.

---

## Remaining work

### Step 9 — Thread-pool for forks

Port `_run_in_pool` + `_pool_semaphore` from original (`clockblocks/clock.py:793-801`). Real perf win when many short forks happen (scamp does this constantly for note playback). Threads-per-fork (cb2's current approach) is fine for correctness but slow.

**Longer-term, eliminate `fork_unsynchronized` for the common case.** Its main scamp use is high-frequency low-overhead bursts — glissandi, continuous expression/volume curves (many MIDI messages with tiny waits). `Clock.schedule_action()` (leaf callbacks fired directly on the scheduler — no child clock, no per-step context-switch handoff) is the intended replacement: sample the envelope and schedule N leaf sends instead of forking a micro-waiting thread. The pool is a stopgap for genuinely thread-y work. **Benchmark first** — the single-sleeper scheduler may already remove most of the per-clock busy-wait lag that motivated `fork_unsynchronized`.

### Step 10 — SCAMP integration

scamp imports (verify against `scamp/src/scamp/__init__.py`): `Clock`, `TempoEnvelope`, `TempoHistory`, `MetricPhaseTarget`; `wait`, `wait_for_children_to_finish`, `wait_forever`, `fork`, `fork_unsynchronized`, `current_clock`; `TimeStamp`; the full `set_tempo_target` / `apply_*_function` family.

Run `scamp/test/test_examples.py` as the integration test before declaring done. Expect minor output diffs (timing precision); review and regenerate goldens.

### Step 11 — Unit-test coverage checklist

Keep `mock_time.py`'s compression trick. Cover (✓ = a test exists; confirm the full matrix):
- Single-clock wait timing under each timing_policy
- Nested fork (child, grandchild) timing
- Tempo change from owning thread / sibling thread (Step 3) / non-clock thread (Step 8)
- Looping envelope wrap-around, function-defined tempo extension across a wait boundary (Step 1)
- Kill cascades (Step 4) ✓
- Tempo change reschedules a pending `schedule_at` fork (Step 3)
- Fast-forward (Step 6) ✓
- TimeStamp consistency across clocks (Step 7) ✓

### Step 13 — Module-level "act on the current clock" tempo helpers

`utilities.py` already exposes `wait`/`fork`/etc. that grab `current_clock()`. Extend the pattern to tempo:
- curve family: `set_tempo_target(s)` / `set_rate_target(s)` / `set_beat_length_target(s)`, `apply_*_function`, `stop_tempo_loop_or_function`
- instantaneous setters: `tempo`/`rate`/`beat_length` are *properties* (no module-level property), so add `set_tempo(x)`/`set_rate(x)`/`set_beat_length(x)` forwarding to the property setter on `current_clock()`. (Decide during impl whether to add `get_*` readers.)

Each forwarder resolves `current_clock()`, raises `NoActiveClockError` if none. **Document prominently that these act on *this* (the current) clock, not the master** — like `wait()`. `set_tempo(120)` inside a fork changes *that fork's* tempo; to change the master, call the method on the master object.

---

## Other Features to consider

- Bring back wait_precisely_until? Maybe only on longer waits? The busy wait thing could work on the scheduler no? Is there a way of gaining that precision without using busy wait.
- Drive the scheduler's "now" from an external thread rather than wall-clock sleep — e.g. follow Logic / a DAW transport instead of free-running. Scheduler-side change only: replace `sleep_until(t)` with "wait until the external source signals scheduler-time `t`" via an injectable time source (default = monotonic clock). Open questions: startup offset/rate mapping; host tempo changes as continuous mapping vs discrete tick stream. Worth prototyping against MIDI clock / MTC first.


## Done criteria

1. All Step-11 unit tests pass.
2. `scamp/test/test_examples.py` passes with goldens regenerated and diffs reviewed.
3. `cb2/` renamed to `clockblocks/` (atomic swap; old dir → `clockblocks_legacy/` or a tag for one release). Rewrite every `~cb2.…` xref to `~clockblocks.…` and qualify remaining bare cross-module xrefs (see the deferred Documentation-pass item).
4. Old `_WaitKeeper`, `rouse_and_hold`, `_woken_early`, `_synchronization_policy`, and the three busy-wait spin loops are gone.


### Resolved design questions

- **Scheduler-action serialization.** cb2 serializes per master clock (`_wake_and_advance_to_next_wait_call` blocks on `_scheduler_park_condition` until the woken clock reaches its next wait), matching original. This is key to musical coordination.
- **Priority ordering.** Original uses `_priority_counter` to break ties between siblings woken at the same beat (earlier-forked first). cb2's `priority: Tuple[int, ...]` in `QueueEvent` is the right shape; confirm `clock_id` (parent's id + counter) gives the same ordering.
- **Multiple master clocks.** The shared-scheduler singleton (`get_scheduler()`) and `scheduler=` arg are **gone**. Every master mints+starts its own `Scheduler`; children inherit the parent's. A shared scheduler *is* what makes a family — fork from one master to sync; make two masters to run independently (independent timing policies). Two live masters sharing one scheduler used to deadlock at bootstrap anyway.
- **Natural-end master strands its scheduler.** A master whose owning thread falls off the end without `kill()` leaks one parked daemon thread until process exit — but it's now a private per-master scheduler, so it poisons no one. Only bites a master on a non-main thread in a long-lived process; on the main thread, thread-end *is* process exit. **Accepted and documented** (`Clock` docstring: "create a master off the main thread → call `master.kill()` when done"). `weakref.finalize`/`atexit` can't help (the parked frame pins `self`). Opt-in fix is the Step-12 context manager. `run_as_server` is fine — its owning thread is alive, scheduler released and idle, `kill()` tears it down.
- **`schedule_at` with `MetricPhaseTarget`** needs the parent's `beat()` accurate at fork time — Step 2 (lazy beat) handles the foreign-thread case; owning-thread was already correct.


## Out of scope for 1.0

- Network sync / distributed clocks.