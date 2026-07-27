import threading
import time
import unittest

from clockblocks.clock import Clock, ClockState
from clockblocks.exceptions import ClockKilledError, DeadClockError, NoActiveClockError, WrongThreadError
from clockblocks.moment import Moment
from clockblocks.utilities import current_clock, wait


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

    # ---- the owning thread's tag ----

    def test_killing_master_releases_its_owning_thread(self):
        """A dead master is no longer the thread's active clock, so the implicit helpers say so."""
        self.assertIs(current_clock(), self.master)
        self.master.kill()
        self.assertIsNone(current_clock())
        with self.assertRaises(NoActiveClockError):
            wait(0.01)

    def test_no_active_clock_error_explains_that_the_clock_was_killed(self):
        """The bare 'no active clock' message is unhelpful when the clock was killed a line ago."""
        self.master.kill()
        with self.assertRaises(NoActiveClockError) as caught:
            wait(0.01)
        self.assertIn("was killed", str(caught.exception))
        self.assertIn("master", str(caught.exception))

    def test_no_active_clock_error_explains_run_as_server_handover(self):
        self.master.kill()  # this test uses a dedicated server clock instead
        server = Clock(name="served").run_as_server()
        with self.assertRaises(NoActiveClockError) as caught:
            wait(0.01)
        self.assertIn("run_as_server()", str(caught.exception))
        self.assertIn("served", str(caught.exception))
        self.master = server

    def test_no_active_clock_error_stays_bare_on_a_thread_that_never_had_one(self):
        """No note when there's nothing to explain — a plain non-clock thread."""
        caught = []

        def on_plain_thread():
            try:
                wait(0.01)
            except NoActiveClockError as e:
                caught.append(str(e))

        t = threading.Thread(target=on_plain_thread)
        t.start()
        t.join()
        self.assertEqual(len(caught), 1)
        self.assertNotIn("Note:", caught[0])

    def test_killing_master_still_raises_dead_clock_error_through_a_reference(self):
        """The other half of the split: holding a reference to the corpse still gets DeadClockError."""
        self.master.kill()
        with self.assertRaises(DeadClockError):
            self.master.wait(0.01)

    def test_killing_master_does_not_clear_a_newer_masters_tag(self):
        """Killing a master out of order must not steal the tag from whoever owns the thread now."""
        first = self.master
        second = Clock(name="second")  # takes over this thread's tag
        self.assertIs(current_clock(), second)
        first.kill()
        self.assertIs(current_clock(), second, "killing the older master stole the newer one's tag")
        second.kill()
        self.assertIsNone(current_clock())

    def test_killing_a_server_master_releases_the_server_thread(self):
        """run_as_server moves ownership to the background thread; kill must untag *that* thread."""
        self.master.kill()  # this test uses a dedicated server clock instead
        server = Clock(name="server").run_as_server()
        time.sleep(0.1)
        # run_as_server hands ownership to the background thread, so the tag tracks that thread, not ours.
        tagged = server._tagged_thread
        self.assertIsNot(tagged, threading.current_thread())
        self.assertIs(getattr(tagged, '__clock__', None), server)
        server.kill()
        self.assertIsNone(getattr(tagged, '__clock__', None))
        self.master = server  # tearDown's kill() is then a no-op on an already-dead clock

    def test_killing_a_child_leaves_the_masters_tag_alone(self):
        """Cascade is master-only for tagging: a child's tag lives on its own pool worker."""
        child = self.master.fork(lambda: current_clock().wait(1.0))
        self.master.wait(0.02)
        child.kill()
        self.assertIs(current_clock(), self.master, "killing a child disturbed the master's thread tag")

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

    # ---- terminal parks now propagate ClockKilledError ----

    def test_kill_propagates_through_wait_for_children(self):
        """A master parked in wait_for_children_to_finish(), killed from outside, raises
        ClockKilledError (it no longer swallows). A normal return means children really finished."""
        def proc():
            current_clock().wait(5.0)   # long-lived child so the parent actually parks

        self.master.fork(proc)
        killer = threading.Timer(0.05, self.master.kill)
        killer.start()
        with self.assertRaises(ClockKilledError):
            self.master.wait_for_children_to_finish()
        killer.join()
        self.assertIs(self.master._state, ClockState.DEAD)

    # ---- context manager ----

    def test_context_manager_kills_and_stops_scheduler_on_exit(self):
        """Leaving a `with Clock()` block kills the clock and (master being 1:1 with its scheduler)
        stops that scheduler thread."""
        with Clock(name="cm") as c:
            self.assertTrue(c.alive)
            sched = c.scheduler
        self.assertIs(c._state, ClockState.DEAD)
        self.assertFalse(c.alive)
        sched.join(timeout=2)
        self.assertFalse(sched.is_alive(), "scheduler thread should have stopped after __exit__")

    def test_context_manager_suppresses_kill_during_wait(self):
        """An external kill interrupts a wait inside the block; __exit__ suppresses the resulting
        ClockKilledError so nothing escapes the `with`."""
        with Clock(name="cm") as c:
            killer = threading.Timer(0.05, c.kill)
            killer.start()
            c.wait(5.0)   # interrupted by kill -> ClockKilledError -> suppressed by __exit__
        killer.join(timeout=1)
        self.assertIs(c._state, ClockState.DEAD)

    def test_context_manager_propagates_other_exceptions(self):
        """A non-kill exception in the body still propagates, but the clock is killed on the way out."""
        with self.assertRaises(ValueError):
            with Clock(name="cm") as c:
                raise ValueError("boom")
        self.assertIs(c._state, ClockState.DEAD)


if __name__ == "__main__":
    unittest.main()
