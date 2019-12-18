import threading
import time
from threading import Event


def sleep_precisely_until(stop_time, interruption_event: Event = None):
    time_remaining = stop_time - time.time()
    if time_remaining <= 0:
        return
    elif time_remaining < 0.0005:
        # when there's fewer than 500 microseconds left, just burn cpu cycles and hit it exactly
        while time.time() < stop_time and (interruption_event is None or not interruption_event.is_set()):
            pass
    else:
        if interruption_event is not None:
            if interruption_event.wait(timeout=time_remaining / 2):
                return
        else:
            time.sleep(time_remaining / 2)
        sleep_precisely_until(stop_time, interruption_event)


def sleep_precisely(secs, interruption_event: Event = None):
    sleep_precisely_until(time.time() + secs, interruption_event)


def current_clock():
    # utility for getting the clock we are currently using (we attach it to the thread when it's started)
    current_thread = threading.current_thread()
    if not hasattr(current_thread, '__clock__'):
        return None
    return threading.current_thread().__clock__


def wait(dt, units="beats"):
    c = current_clock()
    if c is not None:
        current_clock().wait(dt, units=units)
    else:
        time.sleep(dt)


def fork(process_function, name="", initial_rate=None, initial_tempo=None, initial_beat_length=None,
         args=(), kwargs=None):
    clock = current_clock()
    if clock is None:
        raise Exception("Cannot fork function: there is no running clock.")
    else:
        return clock.fork(process_function, name=name, initial_rate=initial_rate, initial_tempo=initial_tempo,
                          initial_beat_length=initial_beat_length, args=args, kwargs=kwargs)


def fork_unsynchronized(process_function, args=(), kwargs=None):
    clock = current_clock()
    if clock is None:
        threading.Thread(target=process_function, args=args, kwargs=kwargs).start()
    else:
        clock.fork_unsynchronized(process_function, args=args, kwargs=kwargs)


def snap_float_to_nice_decimal(x: float, order_of_magnitude_difference=7):
    """
    If x is near to a nice decimal, this rounds it. E.g., given a number like 8.01399999999999214, we want to round
    it to 8.014. We do this by comparing what we get if we round coarsely to what we get if we round precisely, for
    where place_difference represents how much more precise the precise round is than the course round. If they're the
    same, then we should be rounding.

    :param x: number to snap
    :param order_of_magnitude_difference: how many orders of magnitude we compare rounding across
    :return:
    """
    for first_place in range(0, 17 - order_of_magnitude_difference):
        if round(x, first_place) == round(x, first_place + order_of_magnitude_difference):
            return round(x, first_place)
    return x


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
