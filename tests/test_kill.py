import threading
import unittest

from cb2.clock import Clock, ClockKilledError, DeadClockError, ClockState, WrongThreadError
from cb2.moment import Moment
from cb2.utilities import current_clock


class KillTestCase(unittest.TestCase):
    """
    Real-time tests for Step 4 (kill / DeadClockError / ClockKilledError / state machine).

    Each test runs against a fresh master, which mints its own scheduler. tearDown kills that
    scheduler so its thread doesn't linger between tests.
    """

    def setUp(self):
        self.master = Clock(name="master")

    def tearDown(self):
        # master.kill() ends the family and (master being 1:1 with its scheduler) stops that thread.
        self.master.kill()

    # ---- clock_id ----

    def test_sibling_clock_ids_are_distinct(self):
        """Regression: every sibling under the same parent gets a unique clock_id."""
        ids = []

        def proc():
            ids.append(current_clock().clock_id)
            current_clock().wait(0.001)

        for _ in range(3):
            self.master.fork(proc)
        self.master.wait(0.05)
        self.assertEqual(ids, [(0, 0), (0, 1), (0, 2)])

    # ---- state transitions ----

    def test_state_transitions_for_normal_fork(self):
        observed = {}

        def proc():
            observed["during_run"] = current_clock()._state
            current_clock().wait(0.001)

        child = self.master.fork(proc)
        self.assertIs(child._state, ClockState.PENDING)
        self.master.wait(0.05)
        self.assertIs(observed["during_run"], ClockState.ALIVE)
        self.assertIs(child._state, ClockState.DEAD)

    def test_alive_property(self):
        self.assertTrue(self.master.alive)
        self.master.kill()
        self.assertFalse(self.master.alive)

    # ---- entry-check errors ----

    def test_wait_on_dead_clock_raises_dead_clock_error(self):
        self.master.kill()
        with self.assertRaises(DeadClockError):
            self.master.wait(0.01)

    def test_fork_on_dead_clock_raises_dead_clock_error(self):
        self.master.kill()
        with self.assertRaises(DeadClockError):
            self.master.fork(lambda: None)

    def test_fork_on_pending_clock_raises_dead_clock_error(self):
        """A PENDING clock can't be forked from — it's neither dead nor alive yet."""
        child = self.master.fork(lambda: None, when=Moment.after_beats(1.0))
        self.assertIs(child._state, ClockState.PENDING)
        with self.assertRaises(DeadClockError):
            child.fork(lambda: None)
        child.kill()  # release the pending fork event from the scheduler heap

    # ---- killing children ----

    def test_kill_alive_child(self):
        log = []

        def proc():
            for i in range(100):
                log.append(i)
                current_clock().wait(0.005)
            log.append("finished")

        child = self.master.fork(proc)
        self.master.wait(0.02)  # let it run a few iterations
        self.assertIs(child._state, ClockState.ALIVE)
        child.kill()
        self.master.wait(0.05)
        self.assertIs(child._state, ClockState.DEAD)
        self.assertNotIn("finished", log)
        self.assertNotIn(child, self.master.children())

    def test_kill_pending_child_means_it_never_runs(self):
        ran = []

        def proc():
            ran.append("x")

        child = self.master.fork(proc, when=Moment.after_beats(1.0))
        self.assertIs(child._state, ClockState.PENDING)
        child.kill()
        self.assertIs(child._state, ClockState.DEAD)
        self.master.wait(0.05)
        self.assertEqual(ran, [])

    # ---- cascading kill ----

    def test_kill_cascades_from_master_to_child(self):
        """Killing master from a non-clock thread propagates to descendants and raises
        ClockKilledError in the master's wait."""
        log = []

        def proc():
            for i in range(100):
                log.append(i)
                current_clock().wait(0.005)

        child = self.master.fork(proc)
        # schedule kill from a non-clock thread so it can fire while master is mid-wait
        killer = threading.Timer(0.02, self.master.kill)
        killer.start()
        with self.assertRaises(ClockKilledError):
            self.master.wait(1.0)
        killer.join()
        self.assertIs(self.master._state, ClockState.DEAD)
        self.assertIs(child._state, ClockState.DEAD)

    # ---- thread ownership ----

    def test_wait_from_wrong_thread_raises(self):
        """wait() must be called from the clock's own thread."""
        errors = []

        def alien():
            try:
                self.master.wait(0.01)
            except WrongThreadError as e:
                errors.append(e)

        t = threading.Thread(target=alien)
        t.start()
        t.join(timeout=1.0)
        self.assertEqual(len(errors), 1)

    # ---- normal exit cleanup ----

    def test_normal_fork_exit_does_not_hang_subsequent_wait(self):
        """Regression: when a forked _fork_wrapper exits normally, the scheduler's park on
        _scheduler_park_condition must be released so the master can keep going."""
        log = []

        def proc():
            log.append("ran")
            current_clock().wait(0.001)

        self.master.fork(proc)
        self.master.wait(0.05)
        self.assertEqual(log, ["ran"])
        # if this returns at all, the scheduler is still healthy
        self.master.wait(0.01)


if __name__ == "__main__":
    unittest.main()
