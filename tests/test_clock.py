"""
Single-clock wait timing under each timing_policy.

Like the rest of the suite, this runs real-time by default and compressed when CLOCKBLOCKS_TEST_COMPRESSION is set
(see tests/timing.py) — the family picks up the compressed backend automatically, and we measure in the
scheduler's time domain via timing.elapsed / timing.sleep so the assertions hold at any factor.

timing_policy convention: 0 = absolute (cut the wait to land on the absolute target time, even if behind),
1 = relative (wait the exact per-step delta, letting absolute drift accumulate).
"""
import unittest

from clockblocks.clock import Clock, ClockFamilyOptions
from tests import timing


class ClockTimingPolicyTestCase(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, "master"):
            self.master.kill()

    def _run_policy(self, timing_policy):
        wait_durations = [0.3, 0.3, 0.3, 0.3]    # beats (== scheduler-seconds at tempo 60)
        extra_work = [0.0, 0.25, 0.0, 0.0]       # scheduler-seconds of off-clock work after each wake

        # Expected wake times (scheduler-seconds), using the same blend the scheduler computes.
        expected = []
        t_ideal, t_actual = 0.0, 0.0
        for dur, work in zip(wait_durations, extra_work):
            t_ideal += dur
            relative = dur                       # delta from the previous ideal target
            absolute = t_ideal - t_actual        # delta needed to land on the absolute target
            step = max(0.0, relative * timing_policy + absolute * (1 - timing_policy))
            t_actual += step
            expected.append(t_actual)
            t_actual += work

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
