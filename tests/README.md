# clockblocks tests

**These tests were written automatically by Claude Code and have received little human oversight.**
They were created during the clockblocks 1.0 redesign and exist to help *pinpoint regressions*: 
when something breaks, a focused failing test here narrows down where.

They are **not** the primary safety net. The first line of defense against regressions is the
**human-written golden-output tests in SCAMP** (`scamp/test/test_examples.py`), which exercise the whole
Session → Performance → Score pipeline end-to-end against stored reference output. The tests in this 
directory are an additional, low-effort aid.

Note that Tolerances in the real-time timing tests are deliberately generous, and a few of these tests 
are mildly sensitive to machine load.

## Running

```
cd clockblocks
python -m unittest discover -s tests         # all tests (real time, ~50s)
python -m unittest tests.test_fork           # one module

# Run the whole suite on a compressed clock — much faster, same assertions:
CLOCKBLOCKS_TEST_COMPRESSION=10 python -m unittest discover -s tests   # ~7s
```

## Time compression (`CLOCKBLOCKS_TEST_COMPRESSION`)

By default the suite runs in real time (`time.sleep`-paced, ~50s). Set `CLOCKBLOCKS_TEST_COMPRESSION=<factor>` to
run the *whole* suite on a compressed clock — e.g. `10` runs ~7x faster. How it works:

- The scheduler reads "now" and blocks on its sleep condition entirely through an injectable
  `TimingBackend` (`clockblocks/scheduler.py`). `tests/__init__.py` reads the env var and, when set, points the
  scheduler's default-backend factory at a `CompressedTime(factor)` — so **every** Clock/Scheduler the suite
  builds runs compressed, with no per-test change. Unset (or `1`) = ordinary real time, unchanged behavior.
- A compressed family runs `factor`x faster in real time while its *perceived* schedule is unchanged. Tests
  reason in the scheduler's time domain via `tests/timing.py`: `timing.elapsed(start)` (= `real * factor`)
  and `timing.sleep(scheduler_seconds)` (= `real / factor`), so the assertions hold at any factor.
- **Cross-thread handshakes are not compressed** (real context switches), so a `T`-second wait really costs
  `~T/factor + handshake_overhead`. Large factors make that overhead a big fraction of the shrunken waits,
  loosening achievable timing tolerances. **10x is the validated sweet spot** (stable, ~7x speedup); the
  suite still passes at 20x with widened tolerances, beyond which the timing-policy tests get flaky.
- Two tests run only in real time (`@skipUnless(timing.FACTOR == 1, ...)`) and are *skipped* whenever
  compression is on, because their assertions are only coherent at factor 1:
  - the precise-timing tests, which assert sub-millisecond accuracy that compression can't preserve (the
    handshake floor dominates, so you'd be measuring OS jitter, not the spin logic);
  - the "default backend is real" assertion — the suite-wide override deliberately repoints the default
    backend at `CompressedTime` under compression, so the very thing this test checks is intentionally false.
