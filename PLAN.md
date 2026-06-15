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

- **Step 10.5 — Clock tempo-target / tempo-function bridges.** Added `Clock`-level bridges over `TempoHistory`'s methods (`set_*_target(s)`, `apply_*_function`, `stop_tempo_loop_or_function`), each `@_reschedule_after_tempo_change`-wrapped. `set_*_target(s)` express *when* the target is reached as a single `when: ResolvableMoment` — a `Moment` (`after_beats`/`after_time`/`at_beat`/`at_time`) or a `MetricPhaseTarget` (lands on the next matching metric phase). **No `duration`/`duration_units`/`metric_phase_target`** params: the Moment carries its own beats-vs-time axis, and a `MetricPhaseTarget` passed as `when` subsumes the old `metric_phase_target` (it resolves to the matching beat). Bare numbers are rejected (`to_absolute_moment(..., allow_number=False)`) — no deprecation path. **Plural form (`set_*_targets`) builds the curve left-to-right and may freely mix beats- and time-axis `whens` in one call** (`_apply_targets` helper): each `when` is resolved against the *live* clock at call time (so `after_beats(5)` is beat now+5, never "5 past the previous segment"), then dispatched to the matching singular `TempoHistory` setter with its own axis — only the first segment honours `truncate`, the rest append. A time-axis `when` becomes a beat endpoint via the curve built so far (which is why the build is incremental). The whens must come out strictly increasing in clock-time; `_add_segment`'s existing backwards-guard `ValueError` is re-raised with the offending index. **No `loop` on `set_*_targets`** (dropped) — looping lives in **`apply_tempo_envelope(envelope, truncate, loop)`**, which appends a `TempoEnvelope` (intrinsically a beats-domain beat-length curve, so no axis ambiguity) via `TempoHistory.append_envelope`; `stop_tempo_loop_or_function` ends the loop. A non-final `MetricPhaseTarget` in a plural list resolves against *now*, so pair it with `MetricPhaseTarget.min_duration` to keep it past the prior segment. (`MetricPhaseTarget` gained a `min_duration` param: `resolve()` returns the nearest match ≥ `now + min_duration`; default 0 = old behavior.) Singular bridge helpers are `_resolve_when` (resolves `when` → `(duration, pinned_axis)`) and `_resolve_align_to` (resolves `align_to` against that axis → `(alignment_target, curve_shape)`); the underlying `TempoHistory`/`TempoEnvelope` keep their numeric-duration mechanism untouched. **`align_to` (single-segment, landed 2026-06-12):** `set_*_target(..., curve_shape=None, align_to=None)` pins the endpoint's *free* axis (opposite `when`) and **solves curvature** — a `MetricPhaseTarget` lands on the nearest matching phase ("over 20 s, on a downbeat"), a plain `Moment` lands on an exact coordinate. Fixed `align_to` determines the curvature, so an explicit `curve_shape` is discarded with a `warnings.warn`; a phase `align_to` keeps `curve_shape` as a seed. Same-axis or unreachable → `ValueError` (atomic: `TempoHistory.set_beat_length_target` snapshots/restores `self.segments` on failure). `MetricPhaseTarget.units` is now optional (`None`=infer: beats in the `when` role, the free axis in the `align_to` role; explicit-conflicting → error). At the `TempoHistory` layer `_add_segment`'s `metric_phase_target` → `alignment_target`, with the two single-segment solvers generalized to candidate-coordinate form (`_solve_segment_end_{time,beat}`). **Group/plural `align_to` landed in Step 15.** `apply_*_function` was already Moment-free (keeps `domain_start`/`domain_end` + a beats/time `duration_units` axis selector — no single "when" applies). `Clock.time_in_master()` proxies `master.time()`. Obsoleted legacy APIs raise explanatory errors instead of `AttributeError`: `synchronization_policy` (gone — Step 2 made it moot), `rouse_and_hold`/`release_from_suspension` (→ `while_scheduler_quiescent()`). `timing_policy` + `use_*_timing_policy` **kept as real methods** forwarding to `self.scheduler.timing_policy` (settable only on master). `log_processing_time` deferred.

- **Step 9 — Fork thread pool.** The master owns one **`concurrent.futures.ThreadPoolExecutor(max_workers=pool_size)`** (default 200) backing every `fork` and `fork_unsynchronized` in the family; children reach it via `self.master`. `_run_in_pool` `acquire(blocking=False)`s a `BoundedSemaphore`, `submit`s the target, and releases the slot in a single `add_done_callback` (which also surfaces any task exception in red via `_threadpool_error_callback`, since a Future swallows exceptions until `.result()` is read). On exhaustion it falls back to a raw daemon `Thread` + `logging.warning`. `fork`'s `_start_new_clock` and `Clock.fork_unsynchronized` (which tags its pooled worker `_UNSYNCHRONIZED`) both route through it; `_fork_wrapper` resets `__clock__=None` in a `finally` so an idle worker doesn't pin a dead child. The module-level `fork_unsynchronized` with no active clock still uses `_spawn_unsynchronized` (no pool to draw on). `kill()` on the master calls `shutdown(wait=False, cancel_futures=True)`.
  - **Why ThreadPoolExecutor, not `multiprocessing.pool.ThreadPool`** (the original's choice): `ThreadPool` builds an internal multiprocessing `SimpleQueue` whose 2 `SemLock`s are real OS semaphores — on macOS (spawn start method) the `resource_tracker` reports them as *"2 leaked semaphore objects to clean up at shutdown"* whenever the process is Ctrl-C'd before they're unlinked. This is the long-standing SCAMP macOS wart (scampsters thread "leaked semaphore objects"). `ThreadPoolExecutor` is pure-threading (zero semaphores, verified under a forced-spawn trace) and spawns workers **lazily** up to the cap, so an idle Session costs no threads. See `.claudeConvos/2026-06-08-threadpool-macos-semaphore-leak.md`.
  - **The semaphore cap is load-bearing, not just an optimization:** a forked clock occupies its worker for its whole lifetime (it parks in waits) and the executor's task queue is unbounded — without the cap, a fork submitted once all workers are busy would queue forever and never start, stalling musical time. The cap keeps the queue empty; overflow goes to a raw Thread.
  - **Prewarming** (`prewarm_pool=10` default, master-only, 0 disables): `_prewarm_pool(n)` spins up `n` workers at construction via a `threading.Barrier(n+1)` (each warm-up no-op parks on the barrier, forcing a fresh worker per task; then all release idle). Benchmarks: raw thread launch ~60µs steady-state, warm pooled fork ~17–25µs — so prewarming shaves the one-time ramp, but the real steady-state win is worker *reuse*, not prewarming. Flagged in-code as "unclear it's worth it"; kept small + disableable so an idle clock can stay thread-free. One nuance that argues for a non-zero default: creation cost isn't flat — per-creation timing showed the *first* thread ~145µs and the first ~8 elevated (threading-machinery init, allocator/cache warmup, CPU turbo ramp) before settling to ~50µs, so prewarming front-loads specifically the most expensive creations (still sub-ms total, one-time). Measurement caveat: average-over-N benchmarking hides this; use per-iteration timing + min/median (not mean) to see it.
  - Tests: `test_thread_pool.py` (incl. a regression asserting the pool is a `ThreadPoolExecutor`, plus prewarm count/clamp/zero). **Still open** (the Step-9 longer-term item): benchmark, and replace high-frequency `fork_unsynchronized` curve work with `schedule_action`.

- **Step 12 — Master lifecycle context manager.** `Clock.__enter__`/`__exit__` (`__exit__` kills, suppressing only `ClockKilledError`/`DeadClockError`); `with Session() as s:` for free. Terminal parks (`wait_forever`/`wait_for_children_to_finish`) now **propagate** `ClockKilledError` rather than swallowing it, so every wait variant behaves the same (kill mid-wait raises, caught at `_fork_wrapper` for sub-clocks or `__exit__`/user `try/except` for the master). Rationale: bare-script shutdown is Ctrl-C/process-exit, never `ClockKilledError`; only a *programmatic* `kill()` raises, exactly when an exception is the right signal. `run_as_server`'s loop absorbs the kill so its daemon thread exits quietly. Opt-in — main-thread scripts relying on thread-end teardown stay valid. scamp's `performance.py` call site reviewed, needs no change. Tests pass (86).

- **Step 13 — Module-level "act on the current clock" tempo helpers.** `utilities.py` now exposes module-level forwarders mirroring `wait`/`fork`: instantaneous setters `set_tempo`/`set_rate`/`set_beat_length` (plus symmetric `get_tempo`/`get_rate`/`get_beat_length` readers, since the underlying `tempo`/`rate`/`beat_length` are *properties* with no module-level form); the curve family `set_*_target(s)` (Moment-based `when`, see Step 10.5) / `apply_*_function` / `apply_tempo_envelope`; and `stop_tempo_loop_or_function`. Each goes through `_current_clock_or_raise(caller)` — **no unsynchronized fallback** (a `fork_unsynchronized` thread has no tempo to change), so it raises `NoActiveClockError` off any non-clock thread. Like `wait()`, **they act on *this* (the current) clock, not the master** — documented on every forwarder. All re-exported from `cb2/__init__.py`. Tests added to `test_module_api.py` (current-clock set/get, fork-not-master targeting, `when` must be a Moment not a bare number, mixed beats/time `set_tempo_targets`, backwards-when index in error, metric-phase + `min_duration`, looping `apply_tempo_envelope`, off-thread raise for all 17).

- **Step 15 — Group / plural `align_to` on `set_*_targets`.** Extended single-segment `align_to` to the plural `Clock.set_*_targets` (+ `utilities` forwarders): `align_to=` accepts a single `ResolvableMoment` (align the whole call as one run, sugar for `[None]*(N-1)+[value]`) or a per-segment list of `None`/targets where each non-`None` entry closes and bends the run since the previous alignment. Grouping is driven from **`_apply_targets`** (`clock.py`): it builds every segment incrementally via the singular `TempoHistory` setter (`align_to=None`), accumulates the current run's indices + when-axes, and on a non-`None` entry validates single-axis (length > 1), resolves the target on the run's *free* axis via the shared `_resolve_align_to` (and resolves each `when` via the shared `_resolve_when`), and calls the new **`TempoHistory._align_run(run_segments, alignment_target, free_axis)`**. `_align_run` is the group generalization of `_solve_segment_end_{time,beat}`: TIME-free (run pins beats) holds beats and pushes total time via `_adjust_segments_time_duration` (candidates from `get_nearest_matching_times`); BEATS-free (run pins time) proportionally stretches the run's beats onto each candidate end-beat then re-solves curvature to preserve the run's total time, restoring between candidate attempts. The whole plural build is atomic — `_apply_targets` snapshots `tempo_history.segments` and restores (clearing the conversion caches) on any failure. The `curve_shape`-discarded-with-a-warning rule is run-length-1 only (a one-segment run is the singular solve); for genuine groups the per-segment `curve_shapes` seed the distribution. **Cleanup done:** deleted the dead monolithic `TempoHistory.set_*_targets` plural block and the orphaned `adjust_metric_phase_at_beat`/`adjust_time_at_beat`/`adjust_metric_phase_at_time`/`adjust_beat_at_time` (the last two referenced a nonexistent `get_beat_wait_from_time_wait` — never restored; `_align_run` operates on a known segment slice, so no time→beat inversion is needed) and `MetricPhaseTarget.interpret` (tuple-coercion, now unused), plus the now-unused `Tuple`/`logging` imports. `_adjust_segments_time_duration` is kept (used by `_align_run`). Tests added to `test_module_api.py`: TIME-free whole-run phase align (end beats unchanged), BEATS-free phase align (end time pinned), per-segment list with two independent runs, single value over mixed axes raises, unreachable group align rolls the whole call back. 115 tests pass.

- **Documentation pass — largely done.** Docstrings made user-facing (implementation detail pushed to comments), wording ported from original clockblocks where still applicable, enums + `Scheduler` documented, clock-termination walkthrough moved to a module-level lifecycle comment, GPL-3.0 headers added.
  - **Deferred to the rename:** qualify bare cross-*module* xrefs (`:class:`Clock`` → `:class:`~clockblocks.clock.Clock``). The docs are one combined scamp+clockblocks Sphinx build with no intersphinx, so cross-module links work only when fully qualified. Pair with the `cb2`→`clockblocks` rename, since every `~cb2.…` prefix has to be rewritten to `~clockblocks.…` anyway.

---

## Remaining work

### Step 9 — Thread-pool for forks

**Pool is wired up** (see Completed steps). Remaining is the longer-term direction below.

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

### Step 14 — Precise event timing (monotonic basis + guard-band spin)

Bring back the original's `sleep_precisely_until` precision, but at the *one* place cb2 actually sleeps — the scheduler's STEP 1c wait (`scheduler.py`). The single-sleeper design is what makes this cheap: a busy-wait now burns **at most one core, only during the final approach to an event, only for the guard width**, and benefits *every* clock in the family. (Contrast the original's three per-master spin loops — the thing the redesign set out to kill.)

**14a — Prerequisite: switch the timing basis off `time.time()`.** Independently worth doing. `time.time()` is `CLOCK_REALTIME` — NTP slews it and it can *step backwards*, which lands directly in `_compute_wait_duration`'s absolute term (`_start_time + next_event.t - now`) as a timing glitch. Replace `_start_time` / `now` / `_last_wake_time` and `_compute_wait_duration` with **`time.perf_counter()`** throughout (monotonic, highest-resolution, and the same clock we'll spin on — so the sleep and the bookkeeping share one domain; `Condition.wait(timeout)`'s relative timeout is already measured against the monotonic clock internally, so this also removes the cross-domain skew). `wall_time()` only improves (elapsed duration is exactly monotonic's job; nobody here needs epoch/calendar time, and `TimeStamp` already dropped `wall_time`).

**14b — Guard-band spin.** Master-only `precise_timing` flag (default off) + tunable `spin_guard_duration` (default 500µs, matching the original). In STEP 1c, when `precise_timing` and `wait_duration > spin_guard_duration`: coarse-wait on the condition for `wait_duration - spin_guard_duration` (still wake-early-able by queue changes), `continue`; once `wait_duration <= spin_guard_duration`, **release `_queue_change_condition`** and busy-spin on `perf_counter()` to the deadline, then fall through to STEP 2.

Decisions settled in discussion:
- **Release the lock during the spin (tier 2).** Holding it loses *nothing* — the spin never calls `.wait()` (so no lost-wakeup hazard), and `_execute_event`'s head-recheck (line ~365) + the `_killed` recheck independently keep it correct. But `_queue_change_condition` is the one lock every mutator takes; holding it through the spin freezes the *whole family* (other clocks' forks, unrelated reschedules, even `kill()`) for the guard while burning a core. Releasing it lets everything else run; the line-365 recheck catches any change when we go to pop. No generation counter needed — that's an optional tier-3 refinement (abort the spin mid-guard on a preempting reschedule/kill) worth adding only if sub-500µs preemption latency matters.
- **Gate on `wait_duration > spin_guard_duration`, not "only long waits" per se.** The spin cost is *fixed per event* (= guard width of core burn), so cost-as-fraction-of-core = `spin_guard_duration × event_rate`. For dense events (glissandi, continuous-volume curves) whose gaps fall below the guard, this gate naturally skips the spin (no coarse wait to truncate) instead of pegging a core — and dovetails with the Step-9 plan to move dense curve work onto `schedule_action` leaf callbacks (the spin then lands on the batch boundary, not each sample).
- **Empirical note.** Per-call sleep/`wait(timeout)` overshoot is ~constant in absolute terms (dominated by OS wakeup latency at the moment of waking, independent of duration) — so long waits aren't inherently worse *per call*. The real duration-proportional error is clock-domain skew (14a fixes it) and *accumulated* per-event jitter across many short waits (the absolute-timing policy fixes drift; the spin fixes per-event jitter — complementary).
- **Not the whole floor.** RT scheduling (`SCHED_FIFO`) would lower the jitter floor further without spinning (needs privileges); Windows `Condition.wait` does *not* use the high-res timer path `time.sleep` got in 3.11, so the spin matters more there. Deferred; the guard-band spin is the portable closer.

Tests: extend `mock_time.py`-style timing tests — assert spin engages only under the gate, that a queue change during the coarse phase still wakes early, and that precise mode hits the deadline within a tight tolerance under the real clock.

---

## Other Features to consider

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