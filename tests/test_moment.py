import time
import unittest

from cb2.clock import Clock
from cb2.moment import Moment, to_absolute_moment
from cb2.metric_phase import MetricPhaseTarget
from cb2.enums import DurationUnits


class MomentTestCase(unittest.TestCase):
    """
    Unit + behavioral tests for the 'when' layer: Moment, MetricPhaseTarget.resolve, to_absolute_moment,
    and the wait()/wait_until() front-ends. Fresh master + scheduler per test (see test_fork.py).
    """

    def setUp(self):
        self.master = Clock(name="master")

    def tearDown(self):
        # master.kill() ends the family and (master being 1:1 with its scheduler) stops that thread.
        self.master.kill()

    # ---- Moment resolution (no timing) ----

    def test_absolute_moment_resolves_to_itself(self):
        m = Moment.at_beat(8)
        self.assertIs(m.resolve(self.master), m)
        self.assertFalse(m.relative)
        self.assertEqual(m.value, 8)
        self.assertEqual(m.units, DurationUnits.BEATS)

    def test_relative_moment_pins_to_current_beat(self):
        self.master.wait(0.1)
        resolved = Moment.after_beats(2).resolve(self.master)
        self.assertFalse(resolved.relative)
        self.assertAlmostEqual(resolved.value, self.master.beat() + 2, delta=0.05)

    def test_none_resolves_to_now(self):
        self.master.wait(0.1)
        m = to_absolute_moment(None, self.master)
        self.assertFalse(m.relative)
        self.assertAlmostEqual(m.value, self.master.beat(), delta=0.05)

    def test_bare_number_convention(self):
        # relative_if_number controls how a bare number is read
        rel = to_absolute_moment(3, self.master, relative_if_number=True)
        ab = to_absolute_moment(3, self.master, relative_if_number=False)
        self.assertAlmostEqual(rel.value, self.master.beat() + 3, delta=0.05)
        self.assertEqual(ab.value, 3)

    def test_scheduler_time_matches_clock_conversion(self):
        m = Moment.at_beat(5)
        self.assertAlmostEqual(m.scheduler_time(self.master),
                               self.master.clock_to_scheduler_time(5), delta=1e-9)

    def test_relative_moment_rejects_beat_on_and_scheduler_time(self):
        rel = Moment.after_beats(2)
        with self.assertRaises(ValueError):
            rel.beat_on(self.master)
        with self.assertRaises(ValueError):
            rel.scheduler_time(self.master)

    def test_repr_round_trips_to_constructor(self):
        self.assertEqual(repr(Moment.at_beat(8)), "Moment.at_beat(8)")
        self.assertEqual(repr(Moment.after_beats(2)), "Moment.after_beats(2)")
        self.assertEqual(repr(Moment.at_time(1.5)), "Moment.at_time(1.5)")
        self.assertEqual(repr(Moment.after_time(0.5)), "Moment.after_time(0.5)")

    def test_metric_phase_target_resolves_to_next_matching_beat(self):
        self.master.wait(0.3)
        m = MetricPhaseTarget(0, 1).resolve(self.master)   # next integer beat
        self.assertIsInstance(m, Moment)
        self.assertEqual(m.units, DurationUnits.BEATS)
        self.assertAlmostEqual(m.value, 1.0, delta=0.05)

    def test_metric_phase_target_in_time_resolves_to_next_matching_time(self):
        # tempo 120 => time = beat/2, so a beats- and time-phase diverge: at beat 1.5 (time 0.75),
        # the next whole *beat* is 2.0 but the next whole *second* is 1.0 — proving we used time.
        self.master.tempo = 120
        self.master.wait(1.5)
        m = MetricPhaseTarget(0, 1, units="time").resolve(self.master)
        self.assertEqual(m.units, DurationUnits.TIME)
        self.assertAlmostEqual(m.value, 1.0, delta=0.05)
        self.assertEqual(repr(MetricPhaseTarget(0, 2, units="time")),
                         "MetricPhaseTarget(0, 2, units='time')")

    # ---- wait_until / wait behavioral ----

    def test_wait_until_absolute_beat(self):
        self.master.wait(0.05)
        self.master.wait_until(0.2)
        self.assertAlmostEqual(self.master.beat(), 0.2, delta=0.05)

    def test_wait_until_past_returns_immediately(self):
        self.master.wait(0.1)
        t0 = time.time()
        self.master.wait_until(0.05)          # already in the past
        self.assertLess(time.time() - t0, 0.05, "wait_until(past) should return ~immediately")
        self.assertGreaterEqual(self.master.beat(), 0.1, "must not rewind the clock")

    def test_wait_accepts_moment(self):
        self.master.wait(Moment.after_beats(0.1))
        self.assertAlmostEqual(self.master.beat(), 0.1, delta=0.05)

    def test_wait_accepts_metric_phase_target(self):
        self.master.wait(0.3)
        self.master.wait(MetricPhaseTarget(0, 1))   # advance to the next integer beat
        self.assertAlmostEqual(self.master.beat(), 1.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
