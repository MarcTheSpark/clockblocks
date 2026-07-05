"""
Real-time tests for precise event timing (PLAN.md Step 14): the master-only precise_timing flag and its
guard-band busy-spin in the scheduler's STEP 1c.

These assert behavior, not the sub-100µs precision the spin actually achieves (that's too tight to be
non-flaky on a loaded machine — see the Step-14 convo for the real numbers). We check that precise mode
lands its deadline within a comfortable absolute tolerance, that the coarse phase is still interruptible
by a queue change, that a wait already inside the guard band fires correctly, and that the knobs are
master-only.
"""
import time
import threading
import unittest

from cb2.scheduler import Scheduler
from cb2.clock import Clock
from cb2.exceptions import NotMasterClockError
from tests import timing


@unittest.skipUnless(timing.FACTOR == 1,
                     "precise timing is a real-time precision property; not meaningful under time compression")
class PreciseTimingSchedulerTestCase(unittest.TestCase):
    def tearDown(self):
        if hasattr(self, "sched") and self.sched.is_alive():
            self.sched.kill()
            self.sched.join(timeout=1)

    def test_precise_timing_off_by_default(self):
        self.sched = Scheduler()
        self.assertFalse(self.sched.precise_timing)

    def test_precise_mode_hits_deadline(self):
        """With precise_timing on (absolute policy), an event fires very close to its target time."""
        self.sched = Scheduler(timing_policy=0.0, precise_timing=True)
        self.sched.start()
        fired = threading.Event()
        error = {}
        start = time.time()
        target = 0.3

        def action():
            error["e"] = abs((time.time() - start) - target)
            fired.set()

        self.sched.schedule_action(target, action, metadata="precise")
        self.assertTrue(fired.wait(timeout=2))
        # Spin lands within microseconds in practice; 8 ms is a loose, non-flaky regression bound.
        self.assertLess(error["e"], 0.008, f"precise event missed its deadline by {error['e']*1000:.2f} ms")

    def test_coarse_phase_still_wakes_early_on_reschedule(self):
        """
        Precise mode coarse-waits down to the guard band before spinning, so a far event rescheduled
        sooner must still be picked up promptly (the coarse wait is interruptible by a queue change).
        """
        self.sched = Scheduler(timing_policy=1.0, precise_timing=True)
        self.sched.start()
        fired = threading.Event()
        fired_at = {}
        start = time.time()

        def action():
            fired_at["t"] = time.time() - start
            fired.set()

        self.sched.schedule_action(5.0, action, metadata="far")
        time.sleep(0.2)                  # let it enter the coarse timed wait
        self.sched.reschedule(lambda e: e.metadata == "far", lambda e: 0.3)
        self.assertTrue(fired.wait(timeout=2), "rescheduled event did not fire promptly under precise mode")
        self.assertLess(fired_at["t"], 1.0, f"fired at {fired_at['t']:.3f}s, expected ~0.3s")

    def test_wait_inside_guard_band_fires(self):
        """
        A wait shorter than spin_guard_duration takes the 'already inside the guard band' branch (arm the
        busy wait immediately, no coarse wait). It must still fire on time, not hang.
        """
        self.sched = Scheduler(timing_policy=0.0, precise_timing=True, spin_guard_duration=0.5)
        self.sched.start()
        fired = threading.Event()
        error = {}
        start = time.time()
        target = 0.05                    # < spin_guard_duration, so it spins straight away

        def action():
            error["e"] = abs((time.time() - start) - target)
            fired.set()

        self.sched.schedule_action(target, action)
        self.assertTrue(fired.wait(timeout=2))
        self.assertLess(error["e"], 0.008, f"in-guard event missed its deadline by {error['e']*1000:.2f} ms")


class PreciseTimingClockTestCase(unittest.TestCase):
    def setUp(self):
        self.master = Clock(name="master")

    def tearDown(self):
        self.master.kill()

    def test_knobs_settable_on_master(self):
        self.master.precise_timing = True
        self.master.spin_guard_duration = 0.001
        self.assertTrue(self.master.scheduler.precise_timing)
        self.assertEqual(self.master.scheduler.spin_guard_duration, 0.001)

    def test_knobs_rejected_on_child(self):
        child = self.master.fork(lambda: None)
        with self.assertRaises(NotMasterClockError):
            child.precise_timing = True
        with self.assertRaises(NotMasterClockError):
            child.spin_guard_duration = 0.001

    def test_negative_guard_rejected(self):
        with self.assertRaises(ValueError):
            self.master.spin_guard_duration = -0.001


if __name__ == "__main__":
    unittest.main()
