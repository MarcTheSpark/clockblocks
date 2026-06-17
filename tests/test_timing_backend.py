"""
Unit tests for the TimingBackend seam (scheduler.py): the default real backend, CompressedTime's scaled
now() and timeout-dividing sleep condition, factor validation, and that a Scheduler picks up an injected
backend. The seam is also exercised end-to-end by test_clock.py; these pin its contract directly.
"""
import time
import unittest

from cb2.scheduler import Scheduler, TimingBackend, CompressedTime
from tests import timing


class TimingBackendTestCase(unittest.TestCase):
    def test_default_backend_now_tracks_perf_counter(self):
        backend = TimingBackend()
        # now() is real perf_counter time: a real sleep advances it by ~the same amount.
        t0 = backend.now()
        time.sleep(0.05)
        self.assertAlmostEqual(backend.now() - t0, 0.05, delta=0.02)

    def test_default_sleep_condition_does_not_scale(self):
        backend = TimingBackend()
        cond = backend.get_sleep_condition()
        with cond:
            start = time.perf_counter()
            cond.wait(timeout=0.1)        # no notifier, so it waits the full (real) timeout
            elapsed = time.perf_counter() - start
        self.assertAlmostEqual(elapsed, 0.1, delta=0.04)

    def test_compressed_now_runs_faster_by_factor(self):
        backend = CompressedTime(10)
        t0 = backend.now()
        time.sleep(0.05)                  # 0.05 real seconds
        # now() advances 10x, so ~0.5 compressed seconds elapse.
        self.assertAlmostEqual(backend.now() - t0, 0.5, delta=0.1)

    def test_compressed_sleep_condition_divides_timeout(self):
        backend = CompressedTime(10)
        cond = backend.get_sleep_condition()
        with cond:
            start = time.perf_counter()
            cond.wait(timeout=1.0)        # 1.0 compressed seconds -> ~0.1 real seconds
            elapsed = time.perf_counter() - start
        self.assertAlmostEqual(elapsed, 0.1, delta=0.05)

    def test_compressed_sleep_condition_none_timeout_passes_through(self):
        # A None timeout must stay None (untimed waits — the handshake waits — must not become 0/scaled).
        import threading
        backend = CompressedTime(10)
        cond = backend.get_sleep_condition()
        results = []

        def waiter():
            with cond:
                results.append(cond.wait(timeout=None))   # blocks until notified, not on a timer
        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.1)
        self.assertTrue(t.is_alive(), "wait(None) should still be blocked without a notify")
        with cond:
            cond.notify_all()
        t.join(timeout=1)
        self.assertEqual(results, [True])

    def test_compressed_rejects_non_positive_factor(self):
        for bad in (0, -1, -0.5):
            with self.assertRaises(ValueError):
                CompressedTime(bad)

    @unittest.skipUnless(timing.FACTOR == 1, "suite-wide compression overrides the default backend")
    def test_scheduler_defaults_to_real_backend(self):
        sched = Scheduler()
        self.assertIsInstance(sched._time, TimingBackend)
        self.assertNotIsInstance(sched._time, CompressedTime)

    def test_scheduler_uses_injected_backend(self):
        backend = CompressedTime(10)
        sched = Scheduler(time_backend=backend)
        self.assertIs(sched._time, backend)


if __name__ == "__main__":
    unittest.main()
