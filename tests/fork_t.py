import time
from cb2.clock import Clock
from cb2.moment import Moment
from cb2.utilities import wait, current_clock


c = Clock("MASTER", initial_tempo=20)
start = time.time()


def grandchild():
    while True:
        print(current_clock().name, current_clock().beat(), current_clock().time(), time.time()-start)
        wait(0.25, units="time")


def child():
    current_clock().fork(grandchild, initial_rate=2, when=Moment.at_beat(2))
    while True:
        print(current_clock().name, current_clock().beat(), current_clock().time(), time.time()-start)
        wait(1)


c.fork(child, initial_rate=2, when=Moment.at_beat(1))
while True:
    print(current_clock().name, current_clock().beat(), current_clock().time(), time.time()-start)
    wait(1)