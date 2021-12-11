import threading
import time
from threading import Event


def sleep_precisely_until(stop_time: float, interruption_event: Event = None) -> None:
    """
    High-precision sleep until stop_time. Sleeps repeatedly by half of the remaining time until there are fewer than
    500 microseconds left, at which point it implements a busy wait.

    :param stop_time: desired stop time in seconds since the epoch (as is returned by :func:`time.time`)
    :param interruption_event: (optional) an Event used to execute the sleep call. This has the advantage that the
        sleep can be interrupted by calling `set()` on the event.
    """
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


def sleep_precisely(secs: float, interruption_event: Event = None) -> None:
    """
    High-precision sleep for the given number of seconds.

    :param secs: sleep duration
    :param interruption_event: see :func:`sleep_precisely_until`
    """
    sleep_precisely_until(time.time() + secs, interruption_event)


def current_clock():
    # utility for getting the clock we are currently using (we attach it to the thread when it's started)
    current_thread = threading.current_thread()
    if not hasattr(current_thread, '__clock__'):
        return None
    return threading.current_thread().__clock__


def wait(dt: float, units="beats") -> None:
    c = current_clock()
    if c is not None:
        current_clock().wait(dt, units=units)
    else:
        time.sleep(dt)


def snap_float_to_nice_decimal(x: float, order_of_magnitude_difference=7) -> float:
    """
    If x is near to a nice decimal, this rounds it. E.g., given a number like 8.01399999999999214, we want to round
    it to 8.014. We do this by comparing what we get if we round coarsely to what we get if we round precisely,
    where order_of_magnitude_difference represents how much more precise the precise round is than the course round.
    If they're the same, then we should be rounding.

    :param x: number to snap
    :param order_of_magnitude_difference: how many orders of magnitude we compare rounding across
    :return: the rounded value
    """
    for first_place in range(0, 17 - order_of_magnitude_difference):
        if round(x, first_place) == round(x, first_place + order_of_magnitude_difference):
            return round(x, first_place)
    return x