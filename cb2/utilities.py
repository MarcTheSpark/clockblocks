from __future__ import annotations
import threading
import time
from threading import Event
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import clock


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


def current_clock() -> clock.Clock | None:
    # utility for getting the clock we are currently using (we attach it to the thread when it's started)
    current_thread = threading.current_thread()
    return threading.current_thread().__clock__ if hasattr(current_thread, '__clock__') else None


def wait(dt: float, units="beats") -> None:
    if (c := current_clock()) is not None:
        c.wait(dt, units=units)
    else:
        time.sleep(dt)
