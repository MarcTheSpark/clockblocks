"""
Tests for clockblocks.scheduler.Scheduler. Wall-clock paced, but compression-aware: the timing-measuring tests
reason in the scheduler domain via tests/timing (so they honor CLOCKBLOCKS_TEST_COMPRESSION), while the
lock/ordering tests (held, priority) keep real sleeps since they gate on behavior, not measured
time. The timing-policy tolerance scales with the factor (handshake jitter isn't compressed).
"""
import unittest
import time
import threading
from clockblocks.scheduler import Scheduler
from tests import timing


class TestScheduler(unittest.TestCase):
    def tearDown(self):
        if hasattr(self, "sched"):
            self.sched.kill()
            self.sched.join(timeout=1)

    def test_held_blocks_during_execution(self):
        # held() must not return until no event is executing: _execution_lock is held by
        # the run loop for the whole duration of an action, so an external caller blocks until it ends.
        started = threading.Event()
        finished = threading.Event()

        def action():
            started.set()
            time.sleep(0.3)
            finished.set()

        self.sched = Scheduler(timing_policy=0.5)
        self.sched.start()
        self.sched.schedule_action(0.05, action, metadata="long")

        self.assertTrue(started.wait(timeout=1))
        self.assertFalse(finished.is_set())  # action is mid-flight
        with self.sched.held():
            # We only get here once the executing action has finished.
            self.assertTrue(finished.is_set())

    def test_reschedule_shortens_wait(self):
        # An event scheduled far out, then rescheduled sooner, must fire at the new (sooner) time.
        # Exercises the run loop re-peeking the head after a reschedule notify rather than sleeping
        # the stale, longer duration. (The lost-wakeup window itself is timing-dependent; this is an
        # end-to-end sanity check that reschedule-then-fire works promptly.)
        fired_at = {}
        fired = threading.Event()

        self.sched = Scheduler(timing_policy=1.0)
        self.sched.start()
        start = timing.stopwatch()

        def action():
            fired_at["t"] = timing.elapsed(start)
            fired.set()

        self.sched.schedule_action(5.0, action, metadata="far")
        timing.sleep(0.2)  # let the scheduler peek the far event and enter its timed wait (scheduler-seconds)
        self.sched.reschedule(lambda e: e.metadata == "far", lambda e: 0.3)

        self.assertTrue(fired.wait(timeout=2), "event did not fire promptly after reschedule")
        self.assertLess(fired_at["t"], 1.0, f"fired at {fired_at['t']:.3f}s, expected ~0.3s")

    def test_priority_order(self):
        results = []
        event1 = threading.Event()
        event2 = threading.Event()

        def action1():
            results.append("first")
            event1.set()

        def action2():
            results.append("second")
            event2.set()

        self.sched = Scheduler(timing_policy=0.5)
        self.sched.start()
        self.sched.schedule_action(0.2, action2, priority=(1,), metadata="action2")
        self.sched.schedule_action(0.2, action1, priority=(0,), metadata="action1")

        event1.wait(timeout=1)
        event2.wait(timeout=1)
        self.assertEqual(results, ["first", "second"])

    def _test_timing_policy(self, timing_policy):
        # A timing policy of 1 is fully relative: each event fires the exact scheduled delta after the
        #     previous event *fired*, so lateness is never recovered.
        # A timing policy of 0 is fully absolute: each event aims at its exact scheduled time measured
        #     from the scheduler's start, recovering lateness immediately at the cost of a squashed delta.
        # A policy between the two clamps the absolute target to a band around the relative one: a wait may
        #     be compressed to `policy` of its nominal length (or stretched to `1 / policy` of it).
        scheduled_times = [0.05, 0.2, 0.35, 0.6, 0.7]
        action_durations = [0, 0.4, 0, 0.1, 0]  # How long each action takes

        # event_1's 0.4 s action overruns its own 0.15 s gap, so event_2 is already overdue when the
        # scheduler gets to it and fires immediately at 0.6 under every policy. The policies diverge on
        # event_3 (nominal gap 0.25, measured from event_2's *scheduled* time of 0.35):
        #   0.0 absolute: aim at the grid (0.6) -- already due, so event_3 fires back-to-back with
        #                 event_2 at 0.6 as the scheduler catches up, then event_4 lands on grid at 0.7.
        #   0.5 blended:  the wait may be compressed to at most 0.5 * 0.25 = 0.125 s past event_2's
        #                 firing -> 0.725, then 0.825.
        #   1.0 relative: wait the full 0.25 s from event_2's firing -> 0.85, then 0.95. Lateness is
        #                 never recovered (relative timing measures firing-to-firing).
        expected_times = {
            0.0: [0.05, 0.2, 0.6, 0.6,   0.7],
            0.5: [0.05, 0.2, 0.6, 0.725, 0.825],
            1.0: [0.05, 0.2, 0.6, 0.85,  0.95],
        }[timing_policy]

        tested_times = {}

        def action(label, duration=0):
            tested_times[label] = timing.elapsed(start)
            if duration:
                timing.sleep(duration)   # off-clock work, in scheduler-seconds

        self.sched = Scheduler(timing_policy=timing_policy)
        self.sched.start()
        start = timing.stopwatch()

        for i, (sched_time, duration) in enumerate(zip(scheduled_times, action_durations)):
            self.sched.schedule_action(sched_time, lambda i=i, d=duration: action(f"event_{i}", d))

        timing.sleep(max(scheduled_times) + 1)
        self.sched.kill()
        self.sched.join()

        # Achievable precision loosens with the compression factor (handshake jitter isn't compressed),
        # so the tolerance scales: an unchanged 0.01 s at the default factor of 1.
        for i, expected in enumerate(expected_times):
            actual = tested_times[f"event_{i}"]
            self.assertAlmostEqual(actual, expected, delta=0.01 * timing.FACTOR, msg=f"Mismatch for event_{i}")

    def test_timing_policy_relative(self):
        self._test_timing_policy(1.0)

    def test_timing_policy_absolute(self):
        self._test_timing_policy(0.0)

    def test_timing_policy_half(self):
        self._test_timing_policy(0.5)


if __name__ == '__main__':
    unittest.main()
