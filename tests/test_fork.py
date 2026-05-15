import time
import threading
import unittest

from cb2.clock import Clock, ClockState
from cb2.utilities import current_clock
from cb2 import scheduler as scheduler_mod


class ForkTestCase(unittest.TestCase):
    """
    Real-time tests for fork behavior — normal forks, scheduled forks (schedule_at),
    and reschedule-on-tempo-change for pending forks.

    Each test runs against a fresh master + fresh scheduler (see test_kill.py for the
    same isolation pattern and rationale).
    """

    def setUp(self):
        if scheduler_mod._scheduler is not None:
            scheduler_mod._scheduler.kill()
            scheduler_mod._scheduler = None
        self.master = Clock(name="master")

    def tearDown(self):
        if scheduler_mod._scheduler is not None:
            scheduler_mod._scheduler.kill()
            scheduler_mod._scheduler = None

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

    # ---- scheduled fork (numeric beat) ----

    def test_schedule_at_numeric_beat(self):
        """schedule_at=k delays the fork until master beat k."""
        beats_at_fire = []

        def proc():
            beats_at_fire.append(current_clock().parent.beat())

        self.master.fork(proc, schedule_at=0.05)
        self.master.wait(0.1)
        self.assertEqual(len(beats_at_fire), 1)
        # parent beat at fire time should be at or just past schedule_at
        self.assertAlmostEqual(beats_at_fire[0], 0.05, delta=0.02)

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
        t0 = time.time()

        def proc():
            fire_wall_times.append(time.time() - t0)

        self.master.fork(proc, schedule_at=1.0)
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


if __name__ == "__main__":
    unittest.main()