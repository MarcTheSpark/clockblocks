import threading
import time


def sleep_precisely_until(stop_time):
    time_remaining = stop_time - time.time()
    if time_remaining <= 0:
        return
    elif time_remaining < 0.0005:
        # when there's fewer than 500 microseconds left, just burn cpu cycles and hit it exactly
        while time.time() < stop_time:
            pass
    else:
        time.sleep(time_remaining / 2)
        sleep_precisely_until(stop_time)


def sleep_precisely(secs):
    sleep_precisely_until(time.time() + secs)


def current_clock():
    # utility for getting the clock we are currently using (we attach it to the thread when it's started)
    current_thread = threading.current_thread()
    if not hasattr(current_thread, '__clock__'):
        return None
    return threading.current_thread().__clock__


def wait(dt):
    current_clock().wait(dt)
