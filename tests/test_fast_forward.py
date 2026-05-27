import threading
import time
import unittest

from cb2.clock import Clock, NotMasterClockError
from cb2.utilities import current_clock
from cb2 import scheduler as scheduler_mod


class FastForwardTestCase(unittest.TestCase):
    """
    Real-time tests for Step 6 (fast-forwarding). The master runs at the default rate of 1 beat/second,
    so a wait(n) that is *not* fast-forwarded takes ~n wall-clock seconds, while a fast-forwarded one
    returns essentially instantly. We assert on both wall-clock elapsed time and the clock's beat.

    Like the other suites, each test gets a fresh master + scheduler (the scheduler is a module-level
    singleton), torn down explicitly so a parked scheduler can't leak into the next test.
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

    # ---- basic toggle ----

    def test_not_fast_forwarding_by_default(self):
        self.assertFalse(self.master.is_fast_forwarding())

    def test_full_fast_forward_makes_waits_instant(self):
        self.master.fast_forward()
        self.assertTrue(self.master.is_fast_forwarding())
        start = time.time()
        self.master.wait(10)  # 10 beats == 10 real seconds at the default rate
        elapsed = time.time() - start
        self.assertLess(elapsed, 0.5, f"fast-forwarded wait took {elapsed:.3f}s, expected ~instant")
        self.assertAlmostEqual(self.master.beat(), 10, delta=0.05)

    def test_fast_forward_off_resumes_realtime(self):
        # The falling-edge re-anchor: after fast-forward ends, the next wait must run in real time
        # (neither instant nor catching up the whole skipped span).
        self.master.fast_forward()
        self.master.wait(5)  # instant -> beat ~5
        self.master.fast_forward(False)
        self.assertFalse(self.master.is_fast_forwarding())

        start = time.time()
        self.master.wait(1)  # 1 beat == 1 real second
        elapsed = time.time() - start
        self.assertAlmostEqual(elapsed, 1.0, delta=0.3, msg=f"post-FF wait took {elapsed:.3f}s, expected ~1s")
        self.assertAlmostEqual(self.master.beat(), 6, delta=0.05)

    # ---- bounded fast-forward (to a goal) ----

    def test_fast_forward_to_beat_then_resume(self):
        self.master.fast_forward_to_beat(5)
        self.assertTrue(self.master.is_fast_forwarding())

        start = time.time()
        self.master.wait(3)  # entirely within the FF region -> instant
        self.assertLess(time.time() - start, 0.5)
        self.assertAlmostEqual(self.master.beat(), 3, delta=0.05)
        self.assertTrue(self.master.is_fast_forwarding())  # goal (beat 5) not yet reached

        start = time.time()
        self.master.wait(4)  # beat 3 -> 7, crossing the goal at 5: 2 beats FF, then 2 beats real time
        elapsed = time.time() - start
        self.assertAlmostEqual(elapsed, 2.0, delta=0.3, msg=f"crossing wait took {elapsed:.3f}s, expected ~2s")
        self.assertFalse(self.master.is_fast_forwarding())
        self.assertAlmostEqual(self.master.beat(), 7, delta=0.05)

    def test_goal_between_two_waits_compresses_only_up_to_goal(self):
        # Goal at beat 15, with waits landing at beat 10 and beat 20. The beat-10 wait is wholly before
        # the goal -> instant. The beat-20 wait straddles the goal: beats 10->15 are compressed away,
        # beats 15->20 play in real time, so it takes ~5 real seconds. (Mirrors the t=10/t=20, goal=15
        # scheduler-level case.)
        self.master.fast_forward_to_beat(15)

        start = time.time()
        self.master.wait(10)
        self.assertLess(time.time() - start, 0.5, "wait fully before the goal should be instant")
        self.assertTrue(self.master.is_fast_forwarding())

        start = time.time()
        self.master.wait(10)  # beat 10 -> 20, crossing the goal at 15
        elapsed = time.time() - start
        self.assertAlmostEqual(elapsed, 5.0, delta=0.3, msg=f"crossing wait took {elapsed:.3f}s, expected ~5s")
        self.assertFalse(self.master.is_fast_forwarding())
        self.assertAlmostEqual(self.master.beat(), 20, delta=0.05)

    def test_early_wakeup_after_goal_does_not_delay_event(self):
        # Regression for the stale-_was_fast_forwarding bug. After fast-forward reaches a finite goal,
        # the run loop waits out the remaining goal -> event span in real time. If a queue change wakes
        # it early during that wait, the event must STILL fire on schedule. With the bug, the leftover
        # flag caused a second re-anchor on the early wakeup that discarded the already-elapsed wait,
        # pushing the event ~2s late.
        #
        # Arming the bug needs all three: (1) an event BEFORE the goal so _was_fast_forwarding gets set,
        # (2) the next event STRICTLY beyond the goal so there's a real-time tail to wait out, and (3) an
        # early wakeup during that tail. The child below provides (1) and (2); the poke provides (3).
        fired = {}

        def proc():
            c = current_clock()
            for _ in range(4):
                c.wait(1)        # events at beats 1,2,3,4 — before the goal, fast-forwarded (sets the flag)
            c.wait(6)            # next event at beat 10 — beyond the goal, waited out in real time
            fired["beat"] = c.beat()
            fired["elapsed"] = time.time() - start

        self.master.fast_forward_to_beat(5)  # goal at beat 5
        self.master.fork(proc)

        # ~2s into the post-goal real-time tail (beats 5..10 == ~5s), poke the scheduler to force an early
        # wakeup. The poked action is scheduled far in the future so it only notifies the condition — it
        # never competes to become the next event, so the early wakeup itself is the only thing under test.
        sched = scheduler_mod._scheduler
        poke = threading.Timer(2.0, lambda: sched.schedule_action(1000.0, lambda: None, metadata="poke"))

        start = time.time()
        poke.start()
        self.master.wait(11)  # outlive the child
        poke.join()

        self.assertAlmostEqual(fired["elapsed"], 5.0, delta=0.5,
                               msg=f"child reached beat 10 after {fired['elapsed']:.3f}s; an early wakeup "
                                   f"must not delay it past the ~5s real-time tail")
        self.assertAlmostEqual(fired["beat"], 10, delta=0.05)

    def test_fast_forward_to_time_then_resume(self):
        self.master.fast_forward_to_time(4)  # master time is seconds
        start = time.time()
        self.master.wait(4, units="time")  # right up to the goal -> instant
        self.assertLess(time.time() - start, 0.5)
        self.assertAlmostEqual(self.master.time(), 4, delta=0.05)
        self.assertFalse(self.master.is_fast_forwarding())

    def test_fast_forward_in_beats_and_in_time(self):
        self.master.fast_forward_in_beats(3)
        self.assertTrue(self.master.is_fast_forwarding())
        self.master.wait(3)  # instant up to the goal
        self.assertFalse(self.master.is_fast_forwarding())
        self.assertAlmostEqual(self.master.beat(), 3, delta=0.05)

        self.master.fast_forward_in_time(2)
        self.master.wait(2, units="time")
        self.assertAlmostEqual(self.master.time(), 5, delta=0.05)

    # ---- whole-family behavior ----

    def test_fast_forward_advances_children_in_lockstep(self):
        log = []

        def proc():
            for _ in range(5):
                log.append(round(current_clock().beat()))
                current_clock().wait(1)

        self.master.fork(proc)
        self.master.fast_forward()
        start = time.time()
        self.master.wait(5)
        elapsed = time.time() - start
        self.assertLess(elapsed, 1.0, f"fast-forwarded family run took {elapsed:.3f}s")
        self.assertEqual(log, [0, 1, 2, 3, 4])

    def test_is_fast_forwarding_visible_to_children(self):
        observed = {}

        def proc():
            observed["ff"] = current_clock().is_fast_forwarding()
            current_clock().wait(0.001)

        self.master.fast_forward()
        self.master.fork(proc)
        self.master.wait(0.05)
        self.assertTrue(observed.get("ff"))

    # ---- restrictions / validation ----

    def test_fast_forward_only_on_master(self):
        results = {}

        def proc():
            try:
                current_clock().fast_forward()
            except NotMasterClockError:
                results["raised"] = True
            current_clock().wait(0.001)

        self.master.fork(proc)
        self.master.wait(0.05)
        self.assertTrue(results.get("raised"))

    def test_fast_forward_to_past_time_raises(self):
        with self.assertRaises(ValueError):
            self.master.fast_forward_to_time(-1.0)

    def test_fast_forward_to_past_beat_raises(self):
        self.master.fast_forward()
        self.master.wait(5)
        self.master.fast_forward(False)
        with self.assertRaises(ValueError):
            self.master.fast_forward_to_beat(2)  # already at beat ~5


if __name__ == "__main__":
    unittest.main()
