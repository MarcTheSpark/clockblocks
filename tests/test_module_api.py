import time
import threading
import unittest

from cb2.clock import Clock, ClockblocksError, NoActiveClockError, NotMasterClockError, ClockState
from cb2 import utilities
from cb2.utilities import current_clock


class ModuleApiTestCase(unittest.TestCase):
    """
    Tests for the Step-5 surface: Clock.fork_unsynchronized / wait_forever /
    wait_for_children_to_finish / run_as_server, and the module-level wrappers in cb2.utilities
    that delegate to current_clock().

    Same isolation pattern as test_fork.py / test_kill.py: fresh master + fresh scheduler per test.
    The master's __init__ binds it as current_clock() on this (the test) thread.
    """

    def setUp(self):
        self.master = Clock(name="master")

    def tearDown(self):
        # master.kill() ends the family and (master being 1:1 with its scheduler) stops that thread.
        self.master.kill()

    # ---- wait_for_children_to_finish ----

    def test_wait_for_children_to_finish(self):
        done = []

        def child():
            current_clock().wait(0.5)
            done.append("done")

        self.master.fork(child)
        self.master.wait_for_children_to_finish()
        self.assertEqual(done, ["done"])
        self.assertEqual(self.master.children(), ())

    def test_module_wait_for_children_to_finish(self):
        done = []

        def child():
            current_clock().wait(0.5)
            done.append("done")

        # module-level fork + wait, both resolving to self.master via current_clock()
        utilities.fork(child)
        utilities.wait_for_children_to_finish()
        self.assertEqual(done, ["done"])
        self.assertEqual(self.master.children(), ())

    def test_wait_for_children_returns_promptly_after_last_child(self):
        # The child finishes ~0.3s in; wait_for_children_to_finish must return right then, not poll to
        # the next whole beat (~1.0s under the old implementation).
        def child():
            current_clock().wait(0.3)

        self.master.fork(child)
        t0 = time.time()
        self.master.wait_for_children_to_finish()
        elapsed = time.time() - t0
        self.assertLess(elapsed, 0.7,
                        f"wait_for_children_to_finish over-waited ({elapsed:.3f}s) past the last child's end")
        self.assertGreater(elapsed, 0.2, "returned before the child could possibly have finished")
        self.assertEqual(self.master.children(), ())

    def test_kill_of_last_child_releases_wait_for_children(self):
        # Killing the last child from another thread must release a parent blocked in
        # wait_for_children_to_finish() (kill()'s detach runs through the same _detach_child path).
        child = self.master.fork(lambda: current_clock().wait(100))   # would otherwise block ~forever

        def killer():
            time.sleep(0.2)
            child.kill()

        threading.Thread(target=killer).start()
        t0 = time.time()
        self.master.wait_for_children_to_finish()
        elapsed = time.time() - t0
        self.assertLess(elapsed, 1.5,
                        f"kill of last child didn't release wait_for_children_to_finish ({elapsed:.3f}s)")
        self.assertEqual(self.master.children(), ())

    # ---- wait_forever ----

    def test_wait_forever_raises_when_clock_killed(self):
        # wait_forever() unblocks by *raising* ClockKilledError, not returning. For a forked clock the
        # fork wrapper catches it, so the line after wait_forever() is never reached and the child ends DEAD.
        reached_after = threading.Event()

        def proc():
            current_clock().wait_forever()   # blocks until the clock is killed, then raises
            reached_after.set()              # must NOT be reached

        child = self.master.fork(proc)
        self.master.wait(0.05)               # let the child enter wait_forever
        child.kill()
        self.master.wait(0.1)                # give the child thread time to unwind
        self.assertFalse(reached_after.is_set(),
                         "code after wait_forever() ran; it should have raised ClockKilledError instead")
        self.assertIs(child._state, ClockState.DEAD)

    # ---- fork_unsynchronized ----

    def test_fork_unsynchronized_runs_with_no_clock_bound(self):
        result = {}
        done = threading.Event()

        def proc():
            result["clock"] = current_clock()   # no clock is bound to an unsynchronized thread
            done.set()

        self.master.fork_unsynchronized(proc)
        self.assertTrue(done.wait(timeout=3))
        self.assertIsNone(result["clock"])
        # it is not a child clock of the master
        self.assertEqual(self.master.children(), ())

    # ---- module-level fork ----

    def test_module_fork_uses_current_clock(self):
        seen = []

        def proc():
            seen.append(current_clock())

        child = utilities.fork(proc)
        self.assertIs(child.parent, self.master)
        self.master.wait(0.05)
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0], child)

    def test_module_fork_without_clock_raises(self):
        captured = {}

        def runner():
            try:
                utilities.fork(lambda: None)
            except ClockblocksError as e:
                captured["err"] = e

        t = threading.Thread(target=runner)   # fresh thread, no clock bound
        t.start()
        t.join(timeout=2)
        self.assertIn("err", captured)

    # ---- run_as_server ----

    def test_run_as_server_releases_calling_thread_and_keeps_running(self):
        self.assertIs(current_clock(), self.master)
        self.master.run_as_server()
        # the calling (test) thread no longer owns the clock
        self.assertIsNone(current_clock())

        ran = threading.Event()
        # fork directly on the server clock (module-level fork can't be used: this thread has no clock)
        self.master.fork(lambda: ran.set())
        self.assertTrue(ran.wait(timeout=3),
                        "forked work did not run on the backgrounded server clock")

        # clean up the background wait_forever thread so it doesn't linger
        self.master.kill()

    def test_run_as_server_on_child_raises(self):
        child = self.master.fork(lambda: None)
        with self.assertRaises(NotMasterClockError):
            child.run_as_server()

    # ---- strictness: no-clock threads raise instead of silently sleeping ----

    @staticmethod
    def _capture_on_fresh_thread(call):
        """Run `call` on a fresh thread (no clock bound) and return any exception it raised."""
        captured = {}

        def runner():
            try:
                call()
            except BaseException as e:  # noqa: BLE001 - we want whatever it raised
                captured["err"] = e

        t = threading.Thread(target=runner)
        t.start()
        t.join(timeout=2)
        return captured.get("err")

    def test_module_wait_without_clock_raises(self):
        err = self._capture_on_fresh_thread(lambda: utilities.wait(0.01))
        self.assertIsInstance(err, NoActiveClockError)

    def test_module_wait_forever_without_clock_raises(self):
        err = self._capture_on_fresh_thread(utilities.wait_forever)
        self.assertIsInstance(err, NoActiveClockError)

    def test_module_wait_for_children_without_clock_raises(self):
        err = self._capture_on_fresh_thread(utilities.wait_for_children_to_finish)
        self.assertIsInstance(err, NoActiveClockError)

    # ---- unsynchronized threads may use the sleep-based waits ----

    def test_unsynchronized_thread_may_wait(self):
        seen = {}
        done = threading.Event()

        def proc():
            seen["clock"] = current_clock()   # None: not a real clock
            utilities.wait(0.02)              # allowed on an unsynchronized thread (real sleep)
            done.set()

        self.master.fork_unsynchronized(proc)
        self.assertTrue(done.wait(timeout=3),
                        "unsynchronized thread did not complete its sleep-based wait()")
        self.assertIsNone(seen["clock"])

    def test_unsynchronized_thread_cannot_fork(self):
        # a fork_unsynchronized thread has no clock to parent children, so module fork() must raise
        err = threading.Event()
        captured = {}

        def proc():
            try:
                utilities.fork(lambda: None)
            except NoActiveClockError as e:
                captured["err"] = e
            finally:
                err.set()

        self.master.fork_unsynchronized(proc)
        self.assertTrue(err.wait(timeout=3))
        self.assertIn("err", captured)


if __name__ == "__main__":
    unittest.main()
