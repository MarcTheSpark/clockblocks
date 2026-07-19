"""
Lazy timing anchoring: wall-clock time spent before the first event that requires a real wait
(setup between Clock construction and the first wait call — loading resources, forking, typing
in a REPL) must not count against the schedule. The scheduler leaves its wall anchor unplanted
until then; pre-anchor immediate events (each clock's initial wake) fire with no wait and no
anchoring. Without this, setup time shows up as lag: the first wait returns immediately and
subsequent waits run compressed until the lag is absorbed.

Like the rest of the suite, this runs real-time by default and compressed when
CLOCKBLOCKS_TEST_COMPRESSION is set (see tests/timing.py).
"""
import threading
import unittest

from clockblocks.clock import Clock
from clockblocks.scheduler import Scheduler
from tests import timing


class LazyAnchorSchedulerTestCase(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, "sched") and self.sched.is_alive():
            self.sched.kill()
            self.sched.join(timeout=1)

    def test_wall_time_zero_before_anchor(self):
        """Until something requires a real wait, no wall time is accruing."""
        self.sched = Scheduler()
        self.sched.start()
        timing.sleep(0.3)
        self.assertEqual(self.sched.wall_time(), 0.0)
        self.assertEqual(self.sched.lag(), 0.0)

    def test_idle_time_before_first_event_is_free(self):
        """Idle time between scheduler start and the first scheduled event doesn't count against it."""
        # absolute policy: were the anchor planted at thread start, the lag would be corrected
        # immediately and the event below would fire with no wait at all
        self.sched = Scheduler(timing_policy=0.0)
        self.sched.start()
        timing.sleep(0.5)
        fired = threading.Event()
        watch = timing.stopwatch()
        self.sched.schedule_action(0.3, fired.set, metadata="first real wait")
        self.assertTrue(fired.wait(timeout=5))
        elapsed = timing.elapsed(watch)
        self.assertGreater(elapsed, 0.2, "first event fired early: idle time counted against the schedule")
        self.assertLess(elapsed, 0.5)


class LazyAnchorClockTestCase(unittest.TestCase):

    def tearDown(self):
        if hasattr(self, "master"):
            self.master.kill()

    def test_setup_time_before_first_wait_is_free(self):
        """
        The clock's initial wake fires at construction without planting the anchor, so however long
        setup takes, the first wait is a full wait (and the next one isn't policy-compressed).
        """
        self.master = Clock("master")
        timing.sleep(0.5)  # setup: soundfont loading, instrument creation, ...
        for description in ("first", "second"):
            watch = timing.stopwatch()
            self.master.wait(0.3, units="time")
            elapsed = timing.elapsed(watch)
            self.assertGreater(elapsed, 0.2, f"{description} wait cut short: setup time counted as lag")
            self.assertLess(elapsed, 0.5)


if __name__ == '__main__':
    unittest.main()
