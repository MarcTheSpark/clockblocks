import io
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr

from clockblocks.clock import Clock
from clockblocks.clock import ClockFamilyOptions
from clockblocks.utilities import current_clock


class ThreadPoolTestCase(unittest.TestCase):
    """
    Tests for the Step-9 fork thread pool: the master owns a shared ThreadPool that backs every
    fork() in the family, with a bounded semaphore that falls back to a raw Thread (with a warning)
    when the pool is exhausted.

    Fresh master + scheduler per test (see test_fork.py / test_kill.py for the isolation pattern).
    """

    def tearDown(self):
        if getattr(self, "master", None) is not None:
            self.master.kill()

    # ---- pool ownership ----

    def test_pool_lives_on_master_only(self):
        self.master = Clock(name="master")
        self.assertIsInstance(self.master._pool_semaphore, threading.BoundedSemaphore)
        self.assertIsNotNone(self.master._pool)

        child_pool_attrs = []

        def proc():
            c = current_clock()
            # Children don't carry their own pool; they reach it via master.
            child_pool_attrs.append(getattr(c, "_pool", "missing"))
            self.assertIs(c.master, self.master)

        self.master.fork(proc)
        self.master.wait(0.05)
        self.assertEqual(child_pool_attrs, ["missing"])

    def test_pool_is_threading_only_no_semaphores(self):
        """Regression: the pool must be a concurrent.futures.ThreadPoolExecutor, not
        multiprocessing.pool.ThreadPool. The latter builds an internal multiprocessing SimpleQueue whose
        SemLocks are real OS semaphores — on macOS (spawn) those surface as "leaked semaphore objects to
        clean up at shutdown" when the process is interrupted. ThreadPoolExecutor creates none."""
        self.master = Clock(name="master")
        self.assertIsInstance(self.master._pool, ThreadPoolExecutor)

    # ---- prewarming ----

    def test_prewarm_spins_up_workers_at_construction(self):
        """prewarm_pool=N should eagerly create N persistent worker threads before any fork."""
        self.master = Clock(name="master", clock_family_options=ClockFamilyOptions(pool_size=50, prewarm_pool=8))
        self.assertEqual(len(self.master._pool._threads), 8)

    def test_prewarm_zero_creates_no_workers(self):
        """prewarm_pool=0 keeps an idle clock thread-free (no pool workers until a fork demands one)."""
        self.master = Clock(name="master", clock_family_options=ClockFamilyOptions(prewarm_pool=0))
        self.assertEqual(len(self.master._pool._threads), 0)

    def test_prewarm_clamped_to_pool_size(self):
        """A prewarm count larger than pool_size is clamped to pool_size."""
        self.master = Clock(name="master", clock_family_options=ClockFamilyOptions(pool_size=3, prewarm_pool=100))
        self.assertLessEqual(len(self.master._pool._threads), 3)

    # ---- pool reuse ----

    def test_sequential_forks_reuse_pool_workers(self):
        """Many short, non-overlapping forks should draw from the fixed pool of workers rather than
        spawning a fresh thread each time. The pool distributes across its (persistent) workers, so the
        count of distinct worker threads is bounded by pool_size — with a per-fork-thread implementation
        it would instead climb to one per fork."""
        self.master = Clock(name="master", clock_family_options=ClockFamilyOptions(pool_size=3))
        idents = set()

        def proc():
            idents.add(threading.get_ident())
            current_clock().wait(0.001)

        for _ in range(20):
            self.master.fork(proc)
            self.master.wait(0.02)  # let the fork finish and its worker return to the pool

        # 20 forks, but never more than pool_size (3) distinct worker threads.
        self.assertLessEqual(len(idents), 3, f"Expected pooled reuse, saw {len(idents)} distinct threads")
        self.assertNotIn(threading.get_ident(), idents)  # ran off the main thread

    # ---- exhaustion fallback ----

    def test_pool_exhaustion_falls_back_to_thread_with_warning(self):
        """With every pool worker parked in a long fork, a further fork must still run — on a raw
        fallback Thread — and emit a warning rather than block."""
        self.master = Clock(name="master", clock_family_options=ClockFamilyOptions(pool_size=2))
        release = threading.Event()
        started = []

        def blocker():
            started.append(threading.get_ident())
            # hold the worker until released, so it stays occupied
            while not release.is_set():
                current_clock().wait(0.01)

        # occupy both pool workers
        self.master.fork(blocker)
        self.master.fork(blocker)
        self.master.wait(0.05)  # let both enter their wait loops
        self.assertEqual(len(started), 2)

        with self.assertLogs(level="WARNING") as cm:
            self.master.fork(blocker)  # third fork: pool exhausted -> fallback Thread + warning
            self.master.wait(0.05)
        self.assertTrue(any("thread pool" in m for m in cm.output))
        self.assertEqual(len(started), 3)  # the fallback fork still ran

        release.set()
        self.master.wait(0.05)

    # ---- error handling releases pool capacity ----

    def test_pool_task_error_releases_semaphore(self):
        """An exception in a pooled task must release its semaphore slot (via error_callback), so the
        pool doesn't slowly leak capacity. With pool_size=1, repeated erroring tasks should keep using
        the (one) pool worker — i.e. never fall back to a warned raw Thread."""
        self.master = Clock(name="master", clock_family_options=ClockFamilyOptions(pool_size=1))

        def boom():
            raise RuntimeError("intentional")

        # The error_callback prints a red traceback to stderr; swallow it so the test output is clean.
        with redirect_stderr(io.StringIO()):
            # If the semaphore weren't released on error, the first task would exhaust the pool and every
            # subsequent submit would warn + fall back. assertNoLogs confirms that never happens.
            with self.assertNoLogs(level="WARNING"):
                for _ in range(5):
                    self.master.fork(boom)
                    time.sleep(0.05)  # let the task raise and the error_callback release the slot


if __name__ == "__main__":
    unittest.main()
