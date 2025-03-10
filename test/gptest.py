import unittest
import time
import threading

# Assume the scheduler code is defined in a module named scheduler_module.
# For this example, we assume Scheduler, QueueEvent, and Stage are available.

# If necessary, replace the following import with the actual module name:
# from scheduler_module import Scheduler, Stage

# For testing purposes, we'll create a dummy sleep_precisely_until function.
# In your test environment, you may choose to monkey-patch the actual function.
def dummy_sleep_precisely_until(stop_time, wait_event):
    now = time.time()
    remaining = stop_time - now
    if remaining > 0:
        time.sleep(remaining)


# Monkey-patch the sleep_precisely_until in the Scheduler module if needed:
from cb2 import scheduler as scheduler_module
scheduler_module.sleep_precisely_until = dummy_sleep_precisely_until
Scheduler = scheduler_module.Scheduler


class TestScheduler(unittest.TestCase):

    def setUp(self):
        # Create a fresh scheduler instance for each test.
        self.scheduler = Scheduler(timing_policy=1.0)
        self.scheduler.start()
        # Give the scheduler a moment to start
        time.sleep(0.1)

    def tearDown(self):
        # Ensure the scheduler is killed and join the thread.
        self.scheduler.kill()
        # self.scheduler.join()

    def test_immediate_execution(self):
        output = []

        def action():
            output.append("immediate_called")

        # Schedule an event in the past.
        current_time = self.scheduler.time()
        self.scheduler.schedule_action(current_time - 1, action, metadata="immediate")
        time.sleep(0.2)  # Allow time for the action to execute
        self.assertEqual(output, ["immediate_called"])

    def test_scheduled_execution(self):
        output = []

        def action():
            output.append("delayed_called")

        # Schedule an event slightly in the future.
        current_time = self.scheduler.time()
        self.scheduler.schedule_action(current_time + 0.2, action, metadata="delayed")
        time.sleep(0.4)  # Wait long enough for the event to be processed
        self.assertEqual(output, ["delayed_called"])

    def test_hold_and_release(self):
        output = []

        def action():
            output.append("held_called")

        # Hold the scheduler, then schedule an event.
        self.scheduler.hold()
        current_time = self.scheduler.time()
        self.scheduler.schedule_action(current_time + 0.1, action, metadata="held")
        time.sleep(0.2)  # While held, the action should not execute.
        self.assertEqual(output, [])
        # Release the scheduler; the event should then execute.
        self.scheduler.release()
        time.sleep(0.3)
        self.assertEqual(output, ["held_called"])

    def test_priority_order(self):
        output = []

        def action1():
            output.append("first")

        def action2():
            output.append("second")

        current_time = self.scheduler.time()
        # Both actions scheduled for the same time; different priorities.
        self.scheduler.schedule_action(current_time + 0.1, action2, priority=(1,), metadata="action2")
        self.scheduler.schedule_action(current_time + 0.1, action1, priority=(0,), metadata="action1")
        time.sleep(0.3)
        # Action with priority (0,) should execute before (1,).
        self.assertEqual(output, ["first", "second"])

    def test_kill_scheduler(self):
        output = []

        def action():
            output.append("should_not_run")

        # Schedule an event in the future.
        self.scheduler.schedule_action(self.scheduler.time() + 0.2, action, metadata="to_be_killed")
        # Kill the scheduler before the event is due.
        self.scheduler.kill()
        time.sleep(0.3)
        # The action should not have been executed.
        self.assertEqual(output, [])

if __name__ == '__main__':
    unittest.main()
