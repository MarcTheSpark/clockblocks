"""
A forked clock forking a clock.
"""

import time
from clockblocks.clock import Clock, ClockFamilyOptions
from clockblocks.moment import Moment
from clockblocks.utilities import wait, current_clock


start = time.perf_counter()
c = Clock("MASTER", initial_tempo=20, clock_family_options=ClockFamilyOptions(precise_timing=True))

def grandchild():
    while True:
        print(current_clock().name, current_clock().beat(), current_clock().time(), time.perf_counter()-start)
        wait(0.25, units="time")


def child():
    current_clock().fork(grandchild, initial_rate=2, when=Moment.at_beat(2))
    while True:
        print(current_clock().name, current_clock().beat(), current_clock().time(), time.perf_counter()-start)
        wait(1)


c.fork(child, initial_rate=2, when=Moment.at_beat(1))
while True:
    print(current_clock().name, current_clock().beat(), current_clock().time(), time.perf_counter()-start)
    wait(1)