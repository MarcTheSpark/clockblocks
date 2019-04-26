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


def fork(process_function, name="", initial_rate=None, initial_tempo=None, initial_beat_length=None,
         extra_args=(), kwargs=None):
    clock = current_clock()
    if clock is None:
        raise Exception("Cannot fork function: there is no running clock.")
    else:
        clock.fork(process_function, name=name, initial_rate=initial_rate, initial_tempo=initial_tempo,
                   initial_beat_length=initial_beat_length, extra_args=extra_args, kwargs=kwargs)


def fork_unsynchronized(process_function, args=(), kwargs=None):
    clock = current_clock()
    if clock is None:
        threading.Thread(target=process_function, args=args, kwargs=kwargs).start()
    else:
        clock.fork_unsynchronized(process_function, args=args, kwargs=kwargs)


class _PrintColors:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
