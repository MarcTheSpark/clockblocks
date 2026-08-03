# Changelog

> These changelogs are AI-written and human-reviewed, because no one (least of all my wife
> and kids) wants me wasting my precious time meticulously documenting this shit, useful
> though it may be.

All notable user-facing changes to clockblocks are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres (or tries to adhere) to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`Clock.extract_absolute_tempo_envelope()` takes an `end_beat`** (defaulting to the clock's current beat),
  the beat out to which the tempo is extracted. The chain is materialized to that beat on private copies, so
  a still-running clock is never touched.

### Fixed

- **`Clock.extract_absolute_tempo_envelope()` no longer smears instantaneous tempo changes into short
  accelerandi/ritardandi.** It used to sample the absolute tempo curve on a blind uniform grid, so any
  sudden tempo change (e.g. a stepwise tempo envelope) landing inside a sampling window came out as a tiny
  ramp or sharply-curved segment. Sampling is now aligned to the tempo breakpoints of every clock in the
  chain, keeping sudden changes sharp; a chain of constant-tempo ancestors is also handled exactly without
  sampling at all. The extracted curve is more faithful overall (sudden changes stay sudden, and the total
  elapsed time it implies is more accurate).

## [1.2.0] - 2026-08-01

### Added

- **`Clock.terminate_forked_children()`** (and the module-level `terminate_forked_children()` acting on the
  calling thread's clock), which kills this clock's children and their descendants. It's the deliberate
  counterpart to `wait_for_children_to_finish()`: the two are the answers to the same question — a clock has
  run out of its own work while forked children are still going, so hang back for them, or cut them off?
  Calling either one resolves the logged warning described under Fixed below.

- **`Clock.description`**, an optional noun phrase describing what a clock is doing, used in place of its
  label in user-facing messages — currently just that warning. It lets a library built on clockblocks
  report the consequence rather than the machinery — scamp sets it so you read "a note on 'clarinet' that
  was still sounding" rather than an internal clock name.

### Changed

- **`Clock.kill()` now returns only once the clocks it killed have actually finished unwinding.**
  Killing is only a signal: a killed clock runs its `except`/`finally` blocks on its own thread afterwards.
  Previously `kill()` returned before any of that had run, so musical time could move on ahead of it and the
  cleanup would take effect at some arbitrary later beat, differing from run to run. Anything a killed clock
  does on the way out is now finished by the time `kill()` returns. Note that this could be a very long time 
  if the clock is in the middle of a long computation or has an expensive done_callback. But this is the price
  of synchronicity.

- **A `fork(..., done_callback=...)` now runs at the moment its clock ended**, rather than racing the rest
  of the family. It used to run just after the scheduler was let go, so time could move on underneath it: a
  callback doing even a couple of milliseconds of work could observe the family several beats past the end
  of the clock it was reporting on, especially if the clock family was fast-forwarding. The family now remains
  frozen all the way through teardown and the done_callback, regardless of how the clock ended (returning 
  normally, raising, or being killed from inside or outside the clock system).

### Fixed

- **A looping tempo envelope now takes on the new cycle's tempo right at the loop point**. Previously, the loop
  was only extended when going strictly past the end of it, which misses a jump discontinuity there.

- **A fork that outlives its own function no longer strands its children.** When a forked function
  returned while clocks it had forked were still running, the finishing clock detached itself from its
  parent and took the whole live subtree with it. Those descendants kept their queued wakeups but were
  invisible to the rest of the family: an ancestor's `wait_for_children_to_finish()` returned early,
  `kill()` never reached them, and — since their pool threads are non-daemon — the interpreter hung on
  exit instead of terminating.

  A clock's lifetime is now firmly its function's lifetime, however that function ends: unfinished children
  are terminated along with it, and a logged warning names each one, since silently truncating a note or a
  layer is near-impossible to diagnose otherwise. That warning points at both ways out —
  `wait_for_children_to_finish()` to keep the parent alive instead, or `terminate_forked_children()` to cut
  them off deliberately. The same reporting covers the other wind-down paths, so a script that ends (or a
  `with` block that closes) while forked parts are still playing says what it cut off rather than stopping
  silently.

- **An unhandled exception in a forked function no longer freezes the whole clock family.** The fork's
  cleanup was skipped on the error path, leaving the scheduler parked on the dead clock forever, so every
  other clock silently stopped advancing. Cleanup now runs however the function exits — including
  terminating any children the fork outlived, which were otherwise orphaned exactly as described above,
  though quietly here, since the traceback already says why the clock ended. The exception itself is
  reported as before, and an error raised by a `done_callback` is likewise logged rather than allowed to
  interrupt the wind-down.

## [1.1.0] - 2026-07-27

### Added

- **`get_beat()` and `get_time()`**, module-level readers of the current clock's position, completing the
  family alongside `get_tempo()` / `get_rate()` / `get_beat_length()`. Saves the round trip through
  `current_clock().beat`, so a loop can read `while get_beat() < 16:`. Like every other module-level helper,
  they act on *this* clock rather than the master (use `current_clock().master.beat` for that) and raise
  `NoActiveClockError` off a clock thread. They return plain floats, not the callable-float shim that
  `Clock.beat` / `Clock.time` return (see Deprecated below).

### Fixed

- **Setup time before the first wait no longer counts against the schedule.** The scheduler
  anchored its wall-clock reference the moment a `Clock` was constructed, so any time spent
  between construction and the first `wait()` (loading resources, forking, typing in a REPL)
  accrued as lag: the first wait returned immediately, and subsequent waits ran compressed
  (2% fast at the default `timing_policy`) until the lag was absorbed. The anchor is now
  planted at the first event that actually requires waiting, restoring the 0.6.x behavior
  where time doesn't start counting until the first wait.

- **`extract_absolute_tempo_envelope()` no longer hangs on a clock following an open-ended
  tempo function** (e.g. `apply_tempo_function(...)` with no `domain_end`). The extraction
  loop kept auto-extending the followed function while chasing the envelope's end, so
  building a score from such a clock's perspective (scamp's
  `to_score()` on a performance recorded on that clock) never finished. The extraction now
  freezes its copies of the tempo histories, using only what the clock has actually
  materialized — matching 0.6.x behavior.

- **A master clock left running no longer hangs interpreter exit.** Forgetting to `kill()` a master (or a
  scamp `Session`) used to wedge the process: the thread pool's workers are non-daemon and get joined
  during shutdown, but the forked functions running on them typically loop forever, so the join never
  finished. Meanwhile the scheduler — a daemon thread, but daemon threads are only killed at the very end
  of finalization, which was never reached — kept running, spraying `RuntimeError: cannot schedule new
  futures after shutdown` for every fork it attempted, and playback carried on after the program was
  "over". Any still-live master is now killed at the start of shutdown, so the process exits cleanly.
  Killing masters explicitly is still the right thing to do; this just makes forgetting cost a leak until
  exit rather than a hang.

- **A fork that fails to launch no longer leaves a phantom child** attached to its parent. The child clock
  is created before its thread starts, and was only detached by the thread itself, so if the launch raised
  the child stayed in the parent's child list forever — leaking it and stalling any
  `wait_for_children_to_finish()` on that parent.

### Changed

- **The position accessors are now read-only properties: write `clock.beat`, `clock.time`,
  `clock.absolute_rate`, `clock.absolute_tempo`, and `clock.absolute_beat_length` without
  parentheses.** This aligns them with `tempo`/`rate`/`beat_length`, which were already
  properties. The rule throughout the library is now: deterministic state is a property
  (`clock.beat` changes only when an event commits time — never merely because wall time
  has passed, so back-to-back reads agree), while wall-clock *samples* that differ on
  every call remain methods (`projected_beat()`, `projected_time()`, `wall_time()`).

- **Killing a master clock now releases its owning thread.** That thread's `current_clock()` becomes
  `None`, so the module-level helpers (`wait()`, `fork()`, `get_beat()`, ...) raise `NoActiveClockError`
  there instead of `DeadClockError`. This draws a clean line between the two errors: `DeadClockError`
  means "the clock you're holding has died" — still what you get from `master.wait(...)` through a
  reference — while `NoActiveClockError` means "you asked for the implicit clock and there isn't one".
  `run_as_server()` already behaved this way when handing ownership to its background thread; `kill()` was
  the odd one out. Only the master's own thread is affected: a forked child's thread is untagged by its
  own cleanup as before. If you catch `DeadClockError` around a module-level `wait()` following a kill,
  catch `NoActiveClockError` instead.

- **`NoActiveClockError` now explains what became of the thread's clock**, when it knows. Killing a clock
  or handing it to a background thread with `run_as_server()` leaves a note behind, so instead of a bare
  "wait() called on a thread with no active clock" you get "… Note: Clock('master') owned this thread, and
  was killed", or a pointer to fork on the returned object directly after `run_as_server()`. A thread that
  simply never had a clock still gets the plain message.

### Deprecated

- **The old method spelling for the new read-only properties (`clock.beat()`, `clock.time()`, 
  `clock.absolute_rate()`, ...) still works**, for now: The properties now return a float that
  is also callable, but emits a `DeprecationWarning` and will be removed in clockblocks 2.0. 
  Just drop the parentheses to use the new form.


## [1.0.0] - 2026-07-12

A ground-up redesign. Clockblocks is now built around a **single central scheduler**
rather than a tree of clocks that each sleep and wake their own children, which collapses
N nested wait loops into one scheduler loop, replaces the old busy-wait spin loops
outright, and derives every clock's position lazily from one ideal timeline. Behavior
that users depend on — musical coordination, nested tempo, one clock acting at a time —
is preserved, but a number of APIs changed or went away. See *Migrating from 0.6.x*.

If you are not ready to migrate, pin `clockblocks<1.0`. Bug fixes to the old line
continue on the `0.6.x` branch. I mean, maybe.

### Changed

- **Forked functions no longer receive the child clock as an argument.** This is the
  breaking change most existing code will hit. Call `current_clock()` inside the
  function instead.
- **`wait` also accepts a `Moment`.** `wait(4)` and `wait(2, units="time")` work exactly as
  before; in addition, `wait` now takes any `Moment` or `MetricPhaseTarget`
  (`wait(Moment.at_beat(8))`, `wait(Moment.after_time(2))`, `wait(MetricPhaseTarget(0, 4))`), in which case
  `units` is ignored. This is additive — no existing `wait` call changes meaning.
- **`TimeStamp` is rebuilt on scheduler time.** It now wraps a captured scheduler-time plus
  the family master, and projects into any clock via `beat_in_clock(c)` / `time_in_clock(c)`.
  Ordering and equality are on the captured instant alone.
- **Tempo targets are expressed with a `Moment`, not a duration.** `set_tempo_target`,
  `set_rate_target`, `set_beat_length_target` and their plural forms now take a single
  `when` argument describing *when* the target is reached — `Moment.after_beats(4)`,
  `Moment.after_time(2.5)`, `Moment.at_beat(16)`, `Moment.at_time(30)`, or a
  `MetricPhaseTarget`. The old
  `duration` / `duration_units` / `metric_phase_target` parameters are gone, and a bare
  number for `when` is rejected rather than silently interpreted.
- Clocks now derive their position lazily from scheduler time, so `Clock.beat()` and
  `Clock.time()` are live and readable **from any thread**, including foreign callbacks.
  From a foreign thread, use the `with clock.hold_scheduler()` context manager to ensure
  that the scheduler is woken, up-to-date and held in suspension.
- Each master clock owns a private `Scheduler`, which handles the coordination and timing
  of the whole family.
- Forked clocks run on a shared `ThreadPoolExecutor` owned by the master, with a certain
  number of threads pre-warmed at creation of the master clock. Among other things this 
  restructuring resolved the long-standing macOS *"leaked semaphore objects"* warning on Ctrl-C.
- **`Clock.kill()` is rebuilt on the scheduler.** It cascades to descendants, cancels their
  pending wakeups *and* pending forks, wakes any clock parked in `wait` (which raises
  `ClockKilledError`), and detaches from the parent; killing the master also tears down the
  family's scheduler and thread pool. It is safe to call from any thread, and killing an
  already-dead clock is a no-op. A `wait` or `fork` on a dead clock raises `DeadClockError`.
- **Precise timing is now opt-in, and no longer blocks the family.** 0.6.x always waited
  "precisely" — halving sleeps down to the last 500µs, then a busy-wait — on every single
  wait. 1.0 rests on the OS wait by default; set `precise_timing=True` on the master to get
  a guard-band spin (`spin_guard_duration`, default 500µs). Unlike the old spin, it releases
  the scheduler's queue lock before spinning, so it never freezes the rest of the clock
  family, it polls for `kill()` so teardown stays prompt, and it is skipped entirely for
  events closer together than the guard band.

### Added

- `Moment` — the unified vocabulary for expressing *when* something happens, constructed via
  its classmethods `Moment.at_beat`, `Moment.at_time`, `Moment.after_beats`,
  `Moment.after_time`. It is the machinery underlying scheduling generally: `wait`,
  `wait_until`, `fork(when=...)`, `schedule_action` and the tempo-target setters all resolve
  a `Moment` (or a `MetricPhaseTarget`, which is also one).
- `wait_until(when, units="beats")` — wait to an *absolute* beat or time, rather than for a
  duration. A convenience for absolute targets given as a bare number;
  `wait(Moment.at_beat(8))` does the same thing. If the target is already past, it returns
  essentially immediately. Available both as `Clock.wait_until` and, like `wait`, as a
  module-level function acting on the current clock.
- `align_to` on the tempo-target setters: pin the endpoint's free axis and let clockblocks
  solve the curvature — e.g. reach a tempo *over 20 seconds, landing on a downbeat*.
- `apply_tempo_envelope(envelope, truncate, loop)` for appending (and looping) a
  `TempoEnvelope`, and `stop_tempo_loop_or_function()` to end it.
- Module-level tempo helpers that act on the current clock, mirroring `wait` / `fork`:
  `set_tempo`, `set_rate`, `set_beat_length` (and `get_*`), the `set_*_target(s)` and
  `apply_*_function` families.
- `schedule_action(...)` — run a callback at a given `Moment` on the scheduler thread,
  without spawning a clock. Suited to high-frequency work such as parameter curves.
- `TimeStampInterval` — the span between two `TimeStamp`s, projectable into any clock in
  the family as either a beat duration or a time duration.
- New exceptions alongside the existing `ClockblocksError` / `ClockKilledError` /
  `DeadClockError`: `WrongThreadError` (calling `wait`/`fork` on a clock from the wrong
  thread), `NoActiveClockError` (a clock-dependent call from a non-clock thread), and
  `NotMasterClockError` (a master-only setting on a child).
- Clocks are context managers: `with Session() as s:` kills the clock family on exit.
  Recommended for any master created off the main thread.
- `hold_scheduler()` — freeze the clock family at the current instant while a foreign
  thread (a MIDI/OSC/keyboard callback) mutates it. Replaces `rouse_and_hold`.
- `ClockFamilyOptions` for family-wide settings, including an injectable `TimingBackend`
  (`CompressedTime`) that lets test suites run a whole clock family fast and deterministically.
- `MetricPhaseTarget.min_duration`, for requiring a phase match to be at least so far away.

### Removed

- `fork_unsynchronized()` and the unsynchronized-thread sentinel. Use a plain
  `threading.Thread` for detached background work that never calls `wait()`.
- `Clock.rouse_and_hold()` / `Clock.release_from_suspension()` → `with clock.hold_scheduler():`
- `Clock.synchronization_policy` — no longer meaningful now that clock positions are
  computed lazily; there is nothing to keep in sync.
- `Clock.time_in_master()` → `clock.master.time()`
- `Clock.wall_time_in_scheduler()` → `clock.scheduler.wall_time()` / `clock.scheduler.lag()`

  The removed `Clock` attributes above raise an explanatory error rather than an
  `AttributeError`, so existing code fails with a pointer to its replacement.

- `sleep_precisely()` / `sleep_precisely_until()` — the old always-on precise sleep. The
  scheduler now owns waiting; see `precise_timing` under *Changed*.
- `WokenEarlyError` — an artifact of the old wake-the-parent protocol, which no longer exists.

### Migrating from 0.6.x

| 0.6.x | 1.0 |
| --- | --- |
| `def melody(clock): ...` + `fork(melody)` | `def melody(): clock = current_clock()` |
| `s.set_tempo_target(120, 4)` | `s.set_tempo_target(120, Moment.after_beats(4))` |
| `s.set_tempo_target(120, 10, duration_units="time")` | `s.set_tempo_target(120, Moment.after_time(10))` |
| `fork_unsynchronized(f)` | `threading.Thread(target=f, daemon=True).start()` |
| `clock.rouse_and_hold()` … `release_from_suspension()` | `with clock.hold_scheduler(): ...` |
| `clock.time_in_master()` | `clock.master.time()` |
| `clock.synchronization_policy = ...` | (remove; no longer needed) |

## Earlier versions

For changes prior to 1.0, see the commit history on the `0.6.x` branch.
