"""
Single-clock wait timing under each timing_policy.

Like the rest of the suite, this runs real-time by default and compressed when CLOCKBLOCKS_TEST_COMPRESSION is set
(see tests/timing.py) — the family picks up the compressed backend automatically, and we measure in the
scheduler's time domain via timing.elapsed / timing.sleep so the assertions hold at any factor.

timing_policy convention: 0 = absolute (cut the wait to land on the absolute target time, even if behind),
1 = relative (wait the exact per-step delta, measured from the previous wake, letting absolute drift
accumulate). Values in between clamp the absolute target to a band around the relative one, so a wait may be
compressed to `policy` of its nominal length (or stretched to `1 / policy` of it) in order to recover drift.
"""
import unittest

from clockblocks.clock import Clock, ClockFamilyOptions
from tests import timing


class ClockTimingPolicyTestCase(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, "master"):
            self.master.kill()

    # Expected wake times (scheduler-seconds) per policy. Steps 0-1 land on the grid. Then 0.9 s of
    # off-clock work overruns the 0.5 s wait, so step 2 is already overdue and fires immediately at 1.9
    # under every policy -- 0.4 s behind the grid. The policies differ in how they recover:
    #   0.0 absolute: step 3 snaps straight back to the grid (2.0) and stays there.
    #   0.5 blended:  step 3's wait compresses to at most 0.5 * 0.5 = 0.25 s -> 1.9 + 0.25 = 2.15; by
    #                 step 4 the residual fits inside the allowable compression band and it rejoins the
    #                 grid at 2.5.
    #   1.0 relative: every wait is exactly 0.5 s from the previous *firing*, so the 0.4 s of lateness
    #                 is never recovered -> 2.4, 2.9, permanently 0.4 s behind.
    # Note the overrun is essential for test differentiation: with off-clock work shorter than the wait
    # nothing ever falls behind, the policy never engages, and all three would agree.
    EXPECTED = {
        0.0: [0.5, 1.0, 1.9, 2.0,  2.5],
        0.5: [0.5, 1.0, 1.9, 2.15, 2.5],
        1.0: [0.5, 1.0, 1.9, 2.4,  2.9],
    }

    def _run_policy(self, timing_policy):
        wait_durations = [0.5] * 5               # beats (== scheduler-seconds at tempo 60)
        extra_work = [0.0, 0.9, 0.0, 0.0, 0.0]   # scheduler-seconds of off-clock work after each wake
        expected = self.EXPECTED[timing_policy]

        self.master = Clock(initial_tempo=60,
                            clock_family_options=ClockFamilyOptions(timing_policy=timing_policy))

        observed = []
        start = timing.stopwatch()
        for dur, work in zip(wait_durations, extra_work):
            self.master.wait(dur)
            observed.append(timing.elapsed(start))   # scheduler-seconds
            if work:
                timing.sleep(work)                   # off-clock delay in scheduler-seconds

        # 0.08 s holds up to ~10x; above that the (uncompressed) handshake jitter is a big enough fraction
        # of the shrunken real waits that the tolerance has to widen proportionally.
        delta = 0.08 * max(1.0, timing.FACTOR / 10)
        for i, (obs, exp) in enumerate(zip(observed, expected)):
            self.assertAlmostEqual(obs, exp, delta=delta,
                                   msg=f"policy={timing_policy} step {i}: woke at {obs:.3f} (scheduler-s), "
                                       f"expected {exp:.3f}")

    def test_absolute_policy(self):
        self._run_policy(0.0)

    def test_relative_policy(self):
        self._run_policy(1.0)

    def test_blended_policy(self):
        self._run_policy(0.5)


if __name__ == "__main__":
    unittest.main()
