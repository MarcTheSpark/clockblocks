import threading
import unittest

from clockblocks.clock import Clock
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
            record["parent_beat"] = current_clock().parent.beat()

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

    # ---- scheduled fork: relative delay (Moment.after_beats) ----

    def test_fork_relative_delay(self):
        """Moment.after_beats(k) forks k beats from now."""
        beats_at_fire = []

        def proc():
            beats_at_fire.append(current_clock().parent.beat())

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
            beats_at_fire.append(current_clock().parent.beat())

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
        child = self.master.fork(lambda: parent_beat_at_start.append(current_clock().parent.beat()),
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
            result["live_beat"] = c.beat()                 # scheduler-derived, under the new tempo
            result["committed_beat"] = c.tempo_history.beat()   # advanced by STEP 4b

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