import threading
import unittest

from cb2.clock import Clock
from cb2.exceptions import DeadClockError
from cb2.moment import Moment
from cb2.utilities import current_clock
from tests import timing


class ScheduleActionTestCase(unittest.TestCase):
    """
    Tests for Clock.schedule_action — leaf callbacks fired on the scheduler thread (no child clock),
    that still ride the fork/wait machinery: tempo changes reschedule them, kill cancels them.

    Same fresh-master + fresh-scheduler isolation as test_fork.py / test_kill.py.
    """

    def setUp(self):
        self.master = Clock(name="master")

    def tearDown(self):
        # master.kill() ends the family and (master being 1:1 with its scheduler) stops that thread.
        self.master.kill()

    # ---- basic firing ----

    def test_action_fires_after_delay(self):
        beats_at_fire = []
        self.master.schedule_action(lambda: beats_at_fire.append(self.master.beat()), when=Moment.after_beats(0.05))
        self.master.wait(0.1)
        self.assertEqual(len(beats_at_fire), 1)
        self.assertAlmostEqual(beats_at_fire[0], 0.05, delta=0.03)

    def test_action_runs_on_scheduler_thread_with_no_clock(self):
        seen = {}

        def act():
            seen["clock"] = current_clock()                 # None: leaf runs on the scheduler thread
            seen["thread"] = threading.current_thread()

        self.master.schedule_action(act, when=Moment.after_beats(0.02))
        self.master.wait(0.05)
        self.assertIsNone(seen.get("clock"))
        self.assertIs(seen.get("thread"), self.master.scheduler)

    def test_schedule_action_rejects_bare_number(self):
        """`when` requires an explicit Moment, since a bare number's meaning would be ambiguous."""
        with self.assertRaises(TypeError):
            self.master.schedule_action(lambda: None, when=4)

    def test_args_and_kwargs_are_forwarded(self):
        got = {}
        self.master.schedule_action(lambda a, b: got.update(a=a, b=b), when=Moment.after_beats(0.02), args=(1,), kwargs={"b": 2})
        self.master.wait(0.05)
        self.assertEqual(got, {"a": 1, "b": 2})

    def test_many_actions_fire_in_order(self):
        fired = []
        for i in range(5):
            self.master.schedule_action(lambda i=i: fired.append(i), when=Moment.after_beats(0.01 * (i + 1)))
        self.master.wait(0.1)
        self.assertEqual(fired, [0, 1, 2, 3, 4])

    # ---- tempo change reschedules a pending action (preserving its musical beat) ----

    def test_tempo_change_reschedules_pending_action(self):
        """Mirror of test_tempo_change_reschedules_pending_fork: a doubled tempo should make the
        action fire around wall=0.5s instead of 1.0s, since its target beat is preserved."""
        fire_wall_times = []
        t0 = timing.stopwatch()
        self.master.schedule_action(lambda: fire_wall_times.append(timing.elapsed(t0)), when=Moment.after_beats(1.0))
        self.master.wait(0.05)
        self.master.tempo = 120
        self.master.wait(1.0)
        self.assertEqual(len(fire_wall_times), 1)
        self.assertLess(fire_wall_times[0], 0.7,
                        f"action fired at wall={fire_wall_times[0]:.3f}s — tempo change didn't reschedule it.")
        self.assertGreater(fire_wall_times[0], 0.3,
                           f"action fired suspiciously early at wall={fire_wall_times[0]:.3f}s.")

    def test_time_locked_action_is_not_rescheduled_by_tempo(self):
        """Contrast with the beat-locked case: a Moment.after_time target stays fixed in *time* under
        a tempo change (the (value, units) metadata preserves units), so doubling the tempo does NOT
        make it fire early — it still lands near wall=1.0s."""
        fire_wall_times = []
        t0 = timing.stopwatch()
        self.master.schedule_action(lambda: fire_wall_times.append(timing.elapsed(t0)),
                                    when=Moment.after_time(1.0))
        self.master.wait(0.05)
        self.master.tempo = 120
        self.master.wait(2.0)   # plenty of beats; ~1s wall at the new tempo
        self.assertEqual(len(fire_wall_times), 1)
        self.assertGreater(fire_wall_times[0], 0.8,
                           f"time-locked action fired early at wall={fire_wall_times[0]:.3f}s.")
        self.assertLess(fire_wall_times[0], 1.3,
                        f"time-locked action fired late at wall={fire_wall_times[0]:.3f}s.")

    # ---- kill cancels a pending action ----

    def test_kill_cancels_pending_action(self):
        fired = []

        def child_proc():
            current_clock().schedule_action(lambda: fired.append("x"), when=Moment.after_beats(0.1))
            current_clock().wait(2.0)   # stay alive until killed

        child = self.master.fork(child_proc)
        self.master.wait(0.02)          # let the child schedule its action, but kill before it fires (beat 0.1)
        child.kill()
        self.master.wait(0.3)           # past beat 0.1 — an uncancelled action would have fired by now
        self.assertEqual(fired, [], "killing the clock should have cancelled its scheduled action")

    # ---- scheduling on a non-ALIVE clock raises ----

    def test_schedule_on_dead_clock_raises(self):
        child = self.master.fork(lambda: None)
        self.master.wait(0.05)          # let the child run and finish -> DEAD
        with self.assertRaises(DeadClockError):
            child.schedule_action(lambda: None, when=Moment.after_beats(0.1))


if __name__ == "__main__":
    unittest.main()