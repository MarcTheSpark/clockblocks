import threading
import time
import unittest
from unittest import mock

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

    # ---- kill() waits for its victims to unwind ----

    def test_kill_returns_only_once_victim_has_unwound(self):
        """
        Killing is only a signal to the victim's thread; the cleanup that releases whatever the clock was
        holding runs over there. kill() must not return before that has happened, or musical time runs on
        and the release is stamped at the wrong beat.
        """
        unwound_at = []

        def victim():
            try:
                current_clock().wait(50)
            except ClockKilledError:
                time.sleep(0.05)                       # cleanup that takes a moment
                unwound_at.append(self.master.beat)
                raise

        child = self.master.fork(victim)
        self.master.wait(0.2)                          # let it get parked in its wait
        kill_beat = self.master.beat
        child.kill()
        # the cleanup has already run by the time kill() returns, and ran at the beat of the kill
        self.assertEqual(len(unwound_at), 1)
        self.assertAlmostEqual(unwound_at[0], kill_beat, delta=0.02)

    def test_kill_waits_for_whole_subtree(self):
        """kill() cascades to descendants, so it waits for the grandchildren too."""
        unwound = []

        def grandchild():
            try:
                current_clock().wait(50)
            except ClockKilledError:
                unwound.append("grandchild")
                raise

        def child():
            c = current_clock()
            c.fork(grandchild)
            c.wait(50)

        sub = self.master.fork(child)
        self.master.wait(0.2)
        sub.kill()
        self.assertIn("grandchild", unwound)

    def test_self_kill_does_not_deadlock(self):
        """
        A clock killing itself is the case that would hang if kill() naively waited for every victim:
        it would be waiting for the very thread it is running on. The calling thread's own inheritance
        line is skipped for exactly this reason.
        """
        finished = threading.Event()

        def suicidal():
            current_clock().kill()      # must return rather than waiting for this thread
            finished.set()

        self.master.fork(suicidal)
        self.master.wait(0.2)
        self.assertTrue(finished.is_set(), "current_clock().kill() blocked waiting for its own thread")

    def test_killing_the_master_from_a_child_does_not_deadlock(self):
        """A child killing the master: the master is skipped (it has no fork wrapper to signal unwind)
        and the child is skipped as the acting clock, so kill() waits on neither and returns promptly."""
        finished = threading.Event()

        def child():
            current_clock().parent.kill()   # parent is the master here
            finished.set()

        self.master.fork(child)
        # the child kills the master out from under us, so our own wait is interrupted
        with self.assertRaises(ClockKilledError):
            self.master.wait(0.2)
        self.assertTrue(finished.wait(timeout=2),
                        "killing the master blocked waiting on the calling thread")

    def test_killing_a_non_master_ancestor_waits_for_its_unwind(self):
        """
        An ancestor runs on its own thread, so — unlike the acting clock itself — kill() can and does wait
        for it. When a clock kills a forked ancestor from underneath itself, that ancestor's cleanup has
        finished by the time kill() returns, exactly as for any other victim.
        """
        record = {}

        def ancestor_layer():
            c = current_clock()
            c.fork(killer)
            try:
                c.wait(50)
            except ClockKilledError:
                time.sleep(0.05)                 # cleanup that takes a moment
                record["ancestor_unwound"] = True
                raise

        def killer():
            c = current_clock()
            c.wait(0.1)
            c.parent.kill()                      # kill the (non-master) layer we were forked from
            record["unwound_when_kill_returned"] = record.get("ancestor_unwound", False)

        self.master.fork(ancestor_layer)
        self.master.wait(0.4)
        # kill() returned only after the ancestor had run its cleanup, not before
        self.assertTrue(record.get("unwound_when_kill_returned"),
                        "kill() returned before the ancestor it killed had finished unwinding")

    def test_killed_non_master_ancestor_unwinds_at_the_kill_beat(self):
        """
        The point of waiting for a killed ancestor: its cleanup lands at the beat it was cut off, not
        wherever a fast-forwarding master has since run to. This needs both halves of the synchronous kill
        — kill() waiting for the ancestor, and kill() *not* releasing the scheduler for the sub-clock that
        did the killing, so time stays frozen while the ancestor tears down.
        """
        record = {}

        def ancestor_layer():
            c = current_clock()
            c.fork(killer)
            try:
                c.wait(50)
            except ClockKilledError:
                time.sleep(0.05)                 # slow cleanup, e.g. a note-off doing I/O
                record["ancestor_cleanup_beat"] = float(self.master.beat)
                raise

        def killer():
            c = current_clock()
            c.wait(1)
            record["kill_beat"] = float(self.master.beat)
            c.parent.kill()

        self.master.fast_forward()
        self.master.fork(ancestor_layer)
        self.master.wait(50)
        self.assertIn("ancestor_cleanup_beat", record)
        # frozen at the kill beat (1.0), not run away toward 50 while the ancestor was still unwinding
        self.assertAlmostEqual(record["ancestor_cleanup_beat"], record["kill_beat"], delta=0.02)

    def test_master_self_kill_is_prompt(self):
        """
        A master has no fork wrapper and so never signals that it has unwound. Waiting on one would burn
        the whole unwind timeout — and `s.kill()` at the end of a script is the single most common kill
        there is, so this would be a five-second stall on nearly every program.
        """
        master = Clock(name="lonely")
        master.fork(lambda: current_clock().wait(50))
        master.wait(0.2)
        start = time.perf_counter()
        master.kill()
        self.assertLess(time.perf_counter() - start, 1.0, "master self-kill waited on its own thread")

    def test_killing_pending_child_is_prompt(self):
        """A clock still inside its start delay has no thread yet, so there is nothing to wait for."""
        child = self.master.fork(lambda: current_clock().wait(50), when=Moment.after_beats(30))
        self.assertIs(child._state, ClockState.PENDING)
        start = time.perf_counter()
        child.kill()
        self.assertLess(time.perf_counter() - start, 1.0, "waited on a clock that had never started")

    def test_slow_victim_is_reported_but_still_waited_for(self):
        """
        A victim slow to unwind gets called out, but kill() does not give up on it. Returning early would
        put us right back where we started: kill() reporting a clock as finished while its thread is still
        running, nondeterministically, on whichever machine happens to be loaded.
        """
        release = threading.Event()
        self.addCleanup(release.set)        # never leave the wedged thread parked, even if this fails
        unwound = []

        def wedged():
            try:
                current_clock().wait(50)
            except ClockKilledError:
                release.wait(timeout=30)    # wedged in cleanup until the timer below lets go
                unwound.append(True)
                raise

        child = self.master.fork(wedged)
        self.master.wait(0.2)
        # let go well after the warning is due, so we can see the warning *and* the wait continuing past it
        timer = threading.Timer(0.6, release.set)
        timer.start()
        self.addCleanup(timer.cancel)

        with mock.patch("clockblocks.clock._UNWIND_WARNING_DELAY", 0.2):
            start = time.perf_counter()
            with self.assertLogs(level="WARNING") as caught:
                child.kill()
            elapsed = time.perf_counter() - start

        self.assertIn("Still waiting", "\n".join(caught.output))
        # kill() didn't give up when it warned: it returned only once the victim had actually unwound
        self.assertGreater(elapsed, 0.5, "kill() abandoned the victim instead of waiting it out")
        self.assertEqual(unwound, [True])


if __name__ == "__main__":
    unittest.main()
