"""
Real-time tests for the lazy tempo-curve extension (PLAN.md Step 1):

  * a looping TempoEnvelope wraps around and keeps driving tempo past its initial domain, and
  * a tempo *function* applied with an open-ended domain extends lazily as the clock waits past the
    already-materialized portion.

Both assert the *timing* the curve implies (integrating beat_length = 60/tempo over the beats waited),
which is what would break if extension stopped at the original domain boundary.
"""
import math
import unittest

from cb2.clock import Clock
from cb2.tempo_envelope import TempoEnvelope
from tests import timing


class TempoEnvelopeTestCase(unittest.TestCase):
    def setUp(self):
        self.master = Clock(name="master", initial_tempo=60)

    def tearDown(self):
        self.master.kill()

    def test_looping_envelope_wraps_around(self):
        """
        A symmetric envelope 60->120->60 over [1, 1] beats, looped (period == 2 beats). The loop wrapping
        means the second period must take the same wall time as the first; if extension stopped at the
        original domain (beat 2), the tempo would freeze at 60 and period two would take 2.0 sec instead.
        We compare period-over-period rather than a hand-integrated absolute, so the assertion is robust
        to the exact segment curve shape.
        """
        env = TempoEnvelope([60, 120, 60], [1, 1])
        self.master.apply_tempo_envelope(env, loop=True)
        self.assertIsNotNone(self.master.tempo_history.follow_func_or_envelope_loop)

        start = timing.stopwatch()
        self.master.wait(2.0)                        # period one
        period_one = timing.elapsed(start)
        self.master.wait(2.0)                        # period two (only possible if the loop wrapped)
        period_two = timing.elapsed(start) - period_one
        self.assertAlmostEqual(period_two, period_one, delta=0.1,
                               msg=f"period two ({period_two:.3f}s) != period one ({period_one:.3f}s) — "
                                   f"loop did not wrap")
        # back near the start of the loop at the (even) period boundary
        self.assertAlmostEqual(self.master.tempo, 60, delta=8)
        self.master.stop_tempo_loop_or_function()

    def test_function_extends_across_wait_boundary(self):
        """
        Apply tempo(b) = 60 + 30*b over an open-ended domain (domain_end=None). Waiting 4 beats forces
        the curve to lazily extend well past any initial chunk. Time = integral_0^4 60/(60+30b) db
        = 2*ln(3) ~= 2.197 sec, and the live tempo should reach 60 + 30*4 = 180.
        """
        self.master.apply_tempo_function(lambda b: 60 + 30 * b, domain_end=None)

        start = timing.stopwatch()
        self.master.wait(4.0)
        elapsed = timing.elapsed(start)

        expected = 2 * math.log(3)                   # ~2.1972
        self.assertAlmostEqual(elapsed, expected, delta=0.2,
                               msg=f"waited {elapsed:.3f}s for 4 beats under a ramping function, "
                                   f"expected ~{expected:.3f}s")
        self.assertAlmostEqual(self.master.tempo, 180, delta=8)
        self.master.stop_tempo_loop_or_function()


if __name__ == "__main__":
    unittest.main()
