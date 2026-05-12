import time
from cb2.clock import Clock
import unittest
from mock_time import get_compressed_time, get_compressed_sleep, get_compressed_event
from unittest.mock import patch


class ClockTestCase(unittest.TestCase):

    @patch("time.sleep", get_compressed_sleep(20))
    @patch("time.time", get_compressed_time(20))
    @patch("threading.Event", get_compressed_event(20))
    def test_clock_timing_policies(self):
        for tp in [0, 0.5, 0.8, 0.9, 0.98, 1.0]:
            with self.subTest(f"Timing Policy={tp}"):
                c = Clock(initial_tempo=120)
                recorded_times_and_wall_times = []
                expected_times = [0.5, 1.4, 2.1, 3.3]
                c.wait(0.5)
                print(c.beat(), c.time(), c.wall_time_in_scheduler())
                time.sleep(0.7)
                print(c.beat(), c.time(), c.wall_time_in_scheduler())
                c.wait(1.0)
                print(c.beat(), c.time(), c.wall_time_in_scheduler())
                break
                # for t in expected_times:
                #     scheduler.schedule_action(t, lambda: recorded_times_and_wall_times.append((scheduler.time(),
                #                                                                                scheduler.wall_time())))
                # time.sleep(1.1)
                # scheduler.release()
                # expected_wall_times = [time.time() - scheduler._start_time]
                # for current_correct_time, next_correct_time in zip(expected_times[:-1], expected_times[1:]):
                #     expected_wall_times.append(max(expected_wall_times[-1] + tp * (next_correct_time - current_correct_time),
                #                                    next_correct_time))
                # time.sleep(2.5)
                # for (t, wt), et, ewt in zip(recorded_times_and_wall_times, expected_times, expected_wall_times):
                #     self.assertEqual(t, et)
                #     self.assertTrue((wt - ewt)/ewt < 0.01)
