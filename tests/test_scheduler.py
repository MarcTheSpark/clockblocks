"""
Real-time tests for cb2.scheduler.Scheduler. These use wall-clock `time.sleep` so they take a few
seconds and are mildly jitter-sensitive. They should eventually be ported to `mock_time.py`'s
compressed-time mocks for determinism — see PLAN.md Step 11.
"""
import unittest
import time
import threading
from cb2.scheduler import Scheduler


class TestScheduler(unittest.TestCase):
    def tearDown(self):
        if hasattr(self, "sched"):
            self.sched.kill()
            self.sched.join(timeout=1)

    def test_hold_and_release(self):
        results = []
        event_executed = threading.Event()

        def action():
            results.append("executed")
            event_executed.set()

        self.sched = Scheduler(timing_policy=0.5)
        self.sched.start()
        with self.sched.held():
            self.sched.schedule_action(0.2, action, metadata="hold_release")
            time.sleep(0.3)
            self.assertEqual(results, [])
        event_executed.wait(timeout=1)
        self.assertEqual(results, ["executed"])

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
            tested_times[label] = time.time() - start
            if duration:
                time.sleep(duration)

        self.sched = Scheduler(timing_policy=timing_policy)
        self.sched.start()
        start = time.time()

        for i, (sched_time, duration) in enumerate(zip(scheduled_times, action_durations)):
            self.sched.schedule_action(sched_time, lambda i=i, d=duration: action(f"event_{i}", d))

        time.sleep(max(scheduled_times) + 1)
        self.sched.kill()
        self.sched.join()

        for i, expected in enumerate(expected_times):
            actual = tested_times[f"event_{i}"]
            self.assertAlmostEqual(actual, expected, delta=0.01, msg=f"Mismatch for event_{i}")

    def test_timing_policy_relative(self):
        self._test_timing_policy(1.0)

    def test_timing_policy_absolute(self):
        self._test_timing_policy(0.0)

    def test_timing_policy_half(self):
        self._test_timing_policy(0.5)


if __name__ == '__main__':
    unittest.main()
