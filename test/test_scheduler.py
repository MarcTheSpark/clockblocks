import time
from cb2.scheduler import Scheduler
from cb2.utilities import sleep_precisely
import unittest
from mock_time import get_compressed_time, get_compressed_sleep, get_compressed_event
from unittest.mock import patch


class SchedulerTestCase(unittest.TestCase):

    @patch("time.sleep", get_compressed_sleep(20))
    @patch("time.time", get_compressed_time(20))
    @patch("threading.Event", get_compressed_event(20))
    def test_timing_policies(self):
        for tp in [0, 0.5, 0.8, 0.9, 0.98, 1.0]:
            with self.subTest(f"Timing Policy={tp}"):
                scheduler = Scheduler(timing_policy=tp)
                scheduler.start()
                recorded_times_and_wall_times = []
                expected_times = [0.5, 1.4, 2.1, 3.3]
                scheduler.hold()
                for t in expected_times:
                    scheduler.schedule_action(t, lambda: recorded_times_and_wall_times.append((scheduler.time(),
                                                                                               scheduler.wall_time())))
                time.sleep(1.1)
                scheduler.release()
                expected_wall_times = [time.time() - scheduler._start_time]
                for current_correct_time, next_correct_time in zip(expected_times[:-1], expected_times[1:]):
                    expected_wall_times.append(max(expected_wall_times[-1] + tp * (next_correct_time - current_correct_time),
                                                   next_correct_time))
                time.sleep(2.5)
                for (t, wt), et, ewt in zip(recorded_times_and_wall_times, expected_times, expected_wall_times):
                    self.assertEqual(t, et)
                    self.assertTrue((wt - ewt)/ewt < 0.01)
                scheduler.kill()

    @patch("time.sleep", get_compressed_sleep(100))
    @patch("time.time", get_compressed_time(100))
    @patch("threading.Event", get_compressed_event(100))
    def test_timing_policies2(self):
        # simple function to print times
        scheduler = Scheduler()
        # scheduler.timing_policy = 0  # absolute timing policy
        # scheduler.timing_policy = 1  # relative
        scheduler.timing_policy = 0.8  # compromise timing policy

        recorded_scheduler_times = []
        recorded_wall_times = []

        def record_times():
            recorded_scheduler_times.append(scheduler.time())
            recorded_wall_times.append(scheduler.wall_time())

        # this is the action that causes the scheduler to fall behind. All the record_times before 2.2 should be on
        # time, and then starting with the action at 3, they should be 0.8 late, a delay which should then change (or
        # not change) over time due to the timing policy.
        scheduler.schedule_action(2.2, lambda: sleep_precisely(1.6))

        # hold the scheduler and schedule a bunch of prints every second
        for t in range(0, 15):
            scheduler.schedule_action(t, record_times)

        expected_wall_times = [0, 1, 2]

        scheduler.start()
        time.sleep(16)
        scheduler.kill()

        print(recorded_scheduler_times)
        print(recorded_wall_times)
        return True

