import unittest

from cb2.clock import Clock
from cb2.time_stamp import TimeStamp
from cb2 import scheduler as scheduler_mod


class TimeStampTestCase(unittest.TestCase):
    """Unit tests for TimeStamp: capture-now then resolve into any clock's frame."""

    def setUp(self):
        if scheduler_mod._scheduler is not None:
            scheduler_mod._scheduler.kill()
            scheduler_mod._scheduler = None
        self.master = Clock(name="master")

    def tearDown(self):
        if scheduler_mod._scheduler is not None:
            scheduler_mod._scheduler.kill()
            scheduler_mod._scheduler = None

    def test_implicit_clock_from_thread(self):
        # master's __init__ binds itself as __clock__ on the test thread, so TimeStamp() picks it up
        ts = TimeStamp()
        self.assertIs(ts._master, self.master)

    def test_master_only_round_trip(self):
        self.master.wait(2.0)
        ts = TimeStamp(self.master)
        self.assertAlmostEqual(ts.beat_in_clock(self.master), 2.0, places=6)
        self.assertAlmostEqual(ts.time_in_clock(self.master), 2.0, places=6)
        self.assertAlmostEqual(ts.time_in_master, 2.0, places=6)

    def test_child_clock_resolution(self):
        events = []

        def child_proc():
            from cb2.utilities import current_clock
            c = current_clock()
            c.wait(0.1)
            events.append(TimeStamp(c))

        self.master.fork(child_proc)
        self.master.wait(0.2)
        self.assertEqual(len(events), 1)
        ts = events[0]
        # at child beat 0.1 (default tempo), scheduler_time ~ 0.1
        self.assertAlmostEqual(ts.beat_in_clock(self.master), 0.1, places=4)
        self.assertAlmostEqual(ts.time_in_master, 0.1, places=4)

    def test_resolution_across_clocks_with_tempo(self):
        # child at tempo 120 (2 beats/sec); master at default tempo 60 (1 beat/sec).
        # After 0.2s of scheduler time, master is at beat 0.2, child is at beat 0.4.
        events = []

        def child_proc():
            from cb2.utilities import current_clock
            c = current_clock()
            c.tempo = 120
            c.wait(0.4)  # 0.2s scheduler time
            events.append((c, TimeStamp(c)))

        self.master.fork(child_proc)
        self.master.wait(0.3)
        child, ts = events[0]
        self.assertAlmostEqual(ts.beat_in_clock(child), 0.4, places=4)
        self.assertAlmostEqual(ts.time_in_clock(child), 0.2, places=4)
        self.assertAlmostEqual(ts.beat_in_clock(self.master), 0.2, places=4)
        self.assertAlmostEqual(ts.time_in_master, 0.2, places=4)

    def test_foreign_family_rejected(self):
        ts = TimeStamp(self.master)
        # Build a second family on a new scheduler instance
        if scheduler_mod._scheduler is not None:
            scheduler_mod._scheduler.kill()
            scheduler_mod._scheduler = None
        other = Clock(name="other")
        with self.assertRaises(ValueError):
            ts.beat_in_clock(other)
        with self.assertRaises(ValueError):
            ts.time_in_clock(other)

    def test_ordering_and_equality(self):
        ts1 = TimeStamp(self.master)
        self.master.wait(0.5)
        ts2 = TimeStamp(self.master)
        self.assertLess(ts1, ts2)
        self.assertNotEqual(ts1, ts2)
        ts1_again = TimeStamp.__new__(TimeStamp)
        ts1_again.scheduler_time = ts1.scheduler_time
        ts1_again._master = ts1._master
        self.assertEqual(ts1, ts1_again)


if __name__ == "__main__":
    unittest.main()
