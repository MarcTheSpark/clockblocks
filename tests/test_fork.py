import threading
import unittest

from clockblocks.clock import Clock
from clockblocks.exceptions import ClockKilledError
from tests import timing
from clockblocks.moment import Moment
from clockblocks.utilities import current_clock


class ForkTestCase(unittest.TestCase):
    """
    Real-time tests for fork behavior — normal forks, scheduled forks (when=),
    and reschedule-on-tempo-change for pending forks.

    Each test runs against a fresh master + fresh scheduler (see test_kill.py for the
    same isolation pattern and rationale).
    """

    def setUp(self):
        self.master = Clock(name="master")

    def tearDown(self):
        # master.kill() ends the family and (master being 1:1 with its scheduler) stops that thread.
        self.master.kill()

    # ---- basic fork ----

    def test_fork_returns_child_clock(self):
        child = self.master.fork(lambda: None)
        self.assertIsInstance(child, Clock)
        self.assertIs(child.parent, self.master)
        self.assertIn(child, self.master.children())

    def test_forked_function_runs(self):
        ran = []

        def proc():
            ran.append("ok")
            current_clock().wait(0.001)

        self.master.fork(proc)
        self.master.wait(0.05)
        self.assertEqual(ran, ["ok"])

    # ---- nested fork (child, grandchild) timing ----

    def test_nested_fork_grandchild_timing(self):
        """
        A grandchild forked under a child (which runs at a different tempo) keeps time relative to its
        own parent's beat, and its waits land at the expected wall time given the compounded tempos.

        master tempo 60 (beat==sec). child runs at tempo 120 (child beat = 0.5 sec). The child forks a
        grandchild that waits 2 (child-)beats == 1.0 sec wall, then records the wall time and the parent
        (child) beat it observed.
        """
        t0 = timing.stopwatch()
        record = {}

        def grandchild():
            current_clock().wait(2.0)            # 2 grandchild beats; grandchild inherits child's tempo (120)
            record["wall"] = timing.elapsed(t0)
            record["parent_beat"] = current_clock().parent.beat

        def child():
            c = current_clock()
            c.tempo = 120
            c.fork(grandchild)
            c.wait(4.0)                          # keep the child (and thus the family) alive long enough

        self.master.fork(child)
        self.master.wait(2.5)                    # 2.5 sec wall — well past the grandchild's 1.0 sec wait

        self.assertIn("wall", record)
        # 2 beats at tempo 120 == 1.0 sec wall
        self.assertAlmostEqual(record["wall"], 1.0, delta=0.1,
                               msg=f"grandchild woke at {record['wall']:.3f}s, expected ~1.0s")
        # the child advanced 2 beats by the time the grandchild woke
        self.assertAlmostEqual(record["parent_beat"], 2.0, delta=0.15)

    # ---- a clock ending takes its unfinished children with it, audibly ----

    def test_unfinished_children_terminated_with_warning(self):
        """
        A forked function that returns while its own children are still running takes them down with
        it, and says so. Leaving them alive instead would orphan the subtree — still holding queued
        wakeups, but detached from the family and so beyond the reach of kill().
        """
        record = {}

        def grandchild():
            current_clock().wait(5.0)                 # far longer than the child
            record["grandchild_finished"] = True       # must never happen

        def child():
            c = current_clock()
            record["grandchild_clock"] = c.fork(grandchild)
            c.wait(0.05)                               # returns long before the grandchild is done

        with self.assertLogs(level="WARNING") as caught:
            self.master.fork(child)
            self.master.wait(0.3)

        self.assertFalse(record.get("grandchild_finished"))
        self.assertFalse(record["grandchild_clock"].alive)
        # nothing left dangling anywhere in the family
        self.assertEqual(self.master.descendants(), ())
        message = "\n".join(caught.output)
        self.assertIn("1 unfinished child", message)
        self.assertIn("grandchild", message)
        self.assertIn("wait_for_children_to_finish", message)

    def test_description_used_in_warning(self):
        """A clock can describe what it's doing, so a library built on clockblocks can say
        "a note that was still sounding" rather than naming an internal clock."""
        def grandchild():
            current_clock().wait(5.0)

        def child():
            c = current_clock()
            c.fork(grandchild).description = "a note on 'clarinet' that was still sounding"
            c.wait(0.05)

        with self.assertLogs(level="WARNING") as caught:
            self.master.fork(child)
            self.master.wait(0.3)

        self.assertIn("a note on 'clarinet' that was still sounding", "\n".join(caught.output))

    def test_multiple_unfinished_children_all_named(self):
        """The warning lists every child it terminated, not just the first."""
        def grandchild():
            current_clock().wait(5.0)

        def child():
            c = current_clock()
            c.fork(grandchild, name="alpha")
            c.fork(grandchild, name="beta")
            c.wait(0.05)

        with self.assertLogs(level="WARNING") as caught:
            self.master.fork(child)
            self.master.wait(0.3)

        message = "\n".join(caught.output)
        self.assertIn("2 unfinished children", message)
        self.assertIn("alpha", message)
        self.assertIn("beta", message)

    def test_terminate_forked_children_suppresses_the_warning(self):
        """
        The other way to answer the question the warning asks: cut the children off on purpose. Nothing
        is left unfinished by the time the clock winds down, so there is nothing to warn about.
        """
        record = {}

        def grandchild():
            try:
                current_clock().wait(50)
            except ClockKilledError:
                record["cut_off_at"] = self.master.beat
                raise

        def child():
            c = current_clock()
            c.fork(grandchild)
            c.wait(0.05)
            c.terminate_forked_children()

        with self.assertNoLogs(level="WARNING"):
            self.master.fork(child)
            self.master.wait(0.05)
            cutoff_beat = self.master.beat
            self.master.wait(0.3)

        # and the child really was terminated, at the moment of the call rather than later
        self.assertIn("cut_off_at", record)
        self.assertAlmostEqual(record["cut_off_at"], cutoff_beat, delta=0.04)

    def test_terminate_forked_children_with_no_children_is_a_noop(self):
        """Safe to call unconditionally, whether or not anything is still running."""
        reached = []

        def child():
            current_clock().wait(0.05)
            current_clock().terminate_forked_children()
            reached.append(True)

        self.master.fork(child)
        self.master.wait(0.2)
        self.assertEqual(reached, [True])

    def test_module_level_terminate_forked_children(self):
        """The module-level spelling, acting on whatever clock the calling thread is running."""
        from clockblocks.utilities import terminate_forked_children

        killed = []

        def grandchild():
            try:
                current_clock().wait(50)
            except ClockKilledError:
                killed.append(True)
                raise

        def child():
            current_clock().fork(grandchild)
            current_clock().wait(0.05)
            terminate_forked_children()

        with self.assertNoLogs(level="WARNING"):
            self.master.fork(child)
            self.master.wait(0.3)

        self.assertEqual(killed, [True])

    def test_no_warning_when_children_finish_first(self):
        """The warning is strictly about truncation: a fork that outlives its children says nothing."""
        def grandchild():
            current_clock().wait(0.05)

        def child():
            c = current_clock()
            c.fork(grandchild)
            c.wait(0.3)                                # outlasts the grandchild

        with self.assertNoLogs(level="WARNING"):
            self.master.fork(child)
            self.master.wait(0.5)

    def test_wait_for_children_to_finish_suppresses_termination(self):
        """The escape hatch the warning names actually works: waiting explicitly lets children finish."""
        record = {}

        def grandchild():
            current_clock().wait(0.2)
            record["grandchild_finished"] = True

        def child():
            c = current_clock()
            c.fork(grandchild)
            c.wait(0.05)
            c.wait_for_children_to_finish()

        with self.assertNoLogs(level="WARNING"):
            self.master.fork(child)
            self.master.wait(0.5)

        self.assertTrue(record.get("grandchild_finished"))

    # ---- a forked function that raises still winds down cleanly ----
    # (these print a red traceback to stderr, which is the reporting under test, not a failure)

    def test_error_in_forked_function_does_not_freeze_family(self):
        """
        An unhandled exception in a forked function must still detach the clock and release the
        scheduler. Otherwise the scheduler stays parked on the dead clock's condition and every other
        clock in the family stops advancing.
        """
        def boom():
            current_clock().wait(0.05)
            raise ValueError("boom")

        child = self.master.fork(boom, name="boom")
        self.master.wait(0.2)           # the master must keep advancing past the child's error

        self.assertAlmostEqual(self.master.beat, 0.2, delta=0.1)
        self.assertFalse(child.alive)
        self.assertNotIn(child, self.master.children())

    def test_error_in_forked_function_terminates_its_children(self):
        """
        The wind-down after an error is a full one: children go too. Terminating them is what keeps the
        subtree reachable — the clock is detached and marked DEAD either way, so children left running
        would be stranded outside the family, holding queued wakeups on non-daemon pool threads where no
        kill() or wait_for_children_to_finish() can ever reach them.

        Quietly, though: the traceback being reported says why the clock ended, and the usual warning's
        advice (wait for the children, or cut them off deliberately) misses the point next to it.
        """
        record = {}

        def grandchild():
            current_clock().wait(50)
            record["grandchild_finished"] = True      # must never happen

        def child():
            c = current_clock()
            record["grandchild_clock"] = c.fork(grandchild)
            c.wait(0.05)
            raise ValueError("boom")

        with self.assertNoLogs(level="WARNING"):
            self.master.fork(child)
            self.master.wait(0.3)

        self.assertFalse(record.get("grandchild_finished"))
        self.assertFalse(record["grandchild_clock"].alive)
        self.assertEqual(self.master.descendants(), ())

    def test_done_callback_fires_when_forked_function_raises(self):
        """The done_callback reports clock termination, so it runs on the error path too."""
        fired = []

        def boom():
            raise ValueError("boom")

        self.master.fork(boom, done_callback=lambda: fired.append(True))
        self.master.wait(0.1)
        self.assertEqual(fired, [True])

    def test_raising_done_callback_is_reported_and_does_not_freeze_family(self):
        """
        A done_callback is user code, so an error in it is logged rather than raised — the way the
        scheduler treats the actions it runs. Letting it throw would skip the rest of the wind-down.
        """
        def child():
            current_clock().wait(0.1)

        def on_done():
            raise ValueError("boom")

        with self.assertLogs(level="ERROR") as caught:
            self.master.fork(child, done_callback=on_done)
            self.master.wait(0.3)

        self.assertAlmostEqual(self.master.beat, 0.3, delta=0.1)
        self.assertIn("done_callback", "\n".join(caught.output))

    # ---- scheduled fork: relative delay (Moment.after_beats) ----

    def test_fork_relative_delay(self):
        """Moment.after_beats(k) forks k beats from now."""
        beats_at_fire = []

        def proc():
            beats_at_fire.append(current_clock().parent.beat)

        self.master.fork(proc, when=Moment.after_beats(0.05))
        self.master.wait(0.1)
        self.assertEqual(len(beats_at_fire), 1)
        self.assertAlmostEqual(beats_at_fire[0], 0.05, delta=0.03)

    # ---- scheduled fork: absolute beat (Moment.at_beat) ----

    def test_fork_at_absolute_beat(self):
        """Moment.at_beat(k) forks at beat k of the parent, regardless of the current beat —
        distinguishing it from Moment.after_beats (which would fire at 0.05 + 0.12)."""
        beats_at_fire = []

        def proc():
            beats_at_fire.append(current_clock().parent.beat)

        self.master.wait(0.05)                          # advance so absolute != relative
        self.master.fork(proc, when=Moment.at_beat(0.12))
        self.master.wait(0.15)
        self.assertEqual(len(beats_at_fire), 1)
        self.assertAlmostEqual(beats_at_fire[0], 0.12, delta=0.03)

    def test_fork_rejects_bare_number(self):
        """A bare number for `when` is ambiguous (relative or absolute?), so fork() requires a Moment."""
        with self.assertRaises(TypeError):
            self.master.fork(lambda: None, when=4)

    # ---- tempo-change reschedules pending forks ----

    def test_tempo_change_reschedules_pending_fork(self):
        """
        When the parent's tempo changes after a fork has been scheduled, the pending
        fork's scheduler-time should be recomputed so it still fires at the same target
        beat — i.e., fires sooner under a faster tempo.

        Setup: master starts at tempo 60 (1 beat = 1 sec). Fork is scheduled at beat 1.0.
        Before it fires, master tempo doubles to 120 (1 beat = 0.5 sec). The fork should
        now fire around wall_time = 0.5 sec, not 1.0 sec.
        """
        fire_wall_times = []
        t0 = timing.stopwatch()

        def proc():
            fire_wall_times.append(timing.elapsed(t0))

        self.master.fork(proc, when=Moment.after_beats(1.0))
        # change tempo before the scheduled fork fires
        self.master.wait(0.05)  # let things settle; barely any beats elapsed
        self.master.tempo = 120
        # now wait long enough that the fork should fire under the *new* tempo
        self.master.wait(1.0)  # 0.5 sec wall
        self.assertEqual(len(fire_wall_times), 1)
        # Under the original tempo it would have fired around wall = 1.0 sec.
        # Under doubled tempo, it should fire around wall = ~0.5 sec
        # (0.05 sec at tempo 60 = 0.05 beats elapsed; remaining 0.95 beats at tempo 120 = 0.475 sec).
        self.assertLess(fire_wall_times[0], 0.7,
                        f"Fork fired at wall={fire_wall_times[0]:.3f}s — tempo change didn't reschedule it.")
        self.assertGreater(fire_wall_times[0], 0.3,
                           f"Fork fired suspiciously early at wall={fire_wall_times[0]:.3f}s.")

    def test_time_based_fork_parent_offset_survives_tempo_change(self):
        """
        Regression: a fork scheduled at an absolute *time* must finalize child.parent_offset from the
        real fire instant, not from a beat delay precomputed at fork()-call time (which goes stale when
        the tempo changes, since a fixed time maps to a different beat under the new tempo).

        Master starts at tempo 60 (beat == time); fork targets time 1.0. Before it fires, tempo doubles
        to 120 at beat ~0.1 (time ~0.1), so time 1.0 now lands at parent beat ~0.1 + 2*0.9 = 1.9 — not
        the 1.0 the old code would have baked in.
        """
        parent_beat_at_start = []
        child = self.master.fork(lambda: parent_beat_at_start.append(current_clock().parent.beat),
                                 when=Moment.at_time(1.0))
        self.master.wait(0.1)        # beat 0.1, time 0.1
        self.master.tempo = 120
        self.master.wait(2.5)        # ~1.25 sec wall; passes time 1.0 so the fork fires

        self.assertEqual(len(parent_beat_at_start), 1)
        # parent_offset reflects the NEW-tempo beat for time 1.0 (~1.9), not the stale ~1.0
        self.assertAlmostEqual(child.parent_offset, 1.9, delta=0.1)
        # and it matches the parent beat the child actually observed at its start
        self.assertAlmostEqual(child.parent_offset, parent_beat_at_start[0], delta=0.05)

    def test_time_based_wait_survives_mid_wait_tempo_change(self):
        """
        Regression (parallels the parent_offset fix): a time-based wait must compute its post-wait beat
        at wake time, not up front. A child waits 1.0 sec of its own time; while it's blocked, the master
        doubles the child's tempo. The wakeup stays time-locked (fires at child-time 1.0) but lands at a
        different beat under the new tempo (~1.9, not 1.0), so an up-front beat would leave tempo_history's
        committed pointer out of sync with the live beat.
        """
        result = {}

        def child_proc():
            c = current_clock()
            c.wait(Moment.after_time(1.0))     # wait 1 sec of child-time
            result["live_beat"] = c.beat                 # scheduler-derived, under the new tempo
            result["committed_beat"] = c.tempo_history.beat   # advanced by STEP 4b

        child = self.master.fork(child_proc)
        self.master.wait(0.1)        # let the child enter its time-wait (child ~beat 0.1)
        child.tempo = 120            # external tempo change on the child, mid-wait
        self.master.wait(2.0)        # let the child's 1-sec time-wait complete

        self.assertIn("live_beat", result)
        # the scenario actually exercised the new tempo (beat 1.0s-of-time lands well past 1.0 beats)
        self.assertGreater(result["live_beat"], 1.5)
        # the committed pointer must match the live beat — i.e. STEP 4b used the post-wait (not stale) beat
        self.assertAlmostEqual(result["live_beat"], result["committed_beat"], delta=0.02)


if __name__ == "__main__":
    unittest.main()