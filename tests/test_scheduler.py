"""
Tests for cb2.scheduler.Scheduler. Wall-clock paced, but compression-aware: the timing-measuring tests
reason in the scheduler domain via tests/timing (so they honor CB2_TEST_COMPRESSION), while the
lock/ordering tests (while_quiescent, priority) keep real sleeps since they gate on behavior, not measured
time. The timing-policy tolerance scales with the factor (handshake jitter isn't compressed).
"""
import unittest
import time
import threading
from cb2.scheduler import Scheduler
from tests import timing


class TestScheduler(unittest.TestCase):
    def tearDown(self):
        if hasattr(self, "sched"):
            self.sched.kill()
            self.sched.join(timeout=1)

    def test_while_quiescent_blocks_during_execution(self):
        # while_quiescent() must not return until no event is executing: _execution_lock is held by
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
        with self.sched.while_quiescent():
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
        # a timing policy of 1 is fully relative (wait for the exact delta between events, even if we are behind)
        # a timing policy of 0 if fully absolute (wait until the exact time when the event should happen, even if we're
        #     behind, leading to a (possibly egregiously) incorrect delta.
        # a timing policy between 0 and 1 is a weighted average of those two extremes.
        scheduled_times = [0.05, 0.2, 0.35, 0.6, 0.7]
        action_durations = [0, 0.4, 0, 0.1, 0]  # How long each action takes

        expected_times = []
        t_ideal, t_actual = 0, 0
        for scheduled_time, action_dur in zip(scheduled_times, action_durations):
            relative_wait_duration = scheduled_time - t_ideal
            absolute_wait_duration = scheduled_time - t_actual
            dt_timing_policy = max(0, relative_wait_duration * timing_policy + absolute_wait_duration * (1 - timing_policy))
            t_ideal = scheduled_time
            t_actual += dt_timing_policy
            expected_times.append(t_actual)
            t_actual += action_dur

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
