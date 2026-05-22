import threading
import time

from cb2.clock import Clock
from cb2.utilities import wait, current_clock
from cb2.scheduler import get_scheduler
from cb2.tempo_envelope import TempoEnvelope
import math
import logging


c = Clock("MASTER", initial_tempo=20)
start = time.time()


def grandchild():
    while True:
        print(current_clock().name, current_clock().beat(), current_clock().time(), time.time()-start)
        wait(0.25, units="time")


def child():
    current_clock().fork(grandchild, initial_rate=2, schedule_at=2)
    while True:
        print(current_clock().name, current_clock().beat(), current_clock().time(), time.time()-start)
        wait(1)


c.fork(child, initial_rate=2, schedule_at=1)
while True:
    print(current_clock().name, current_clock().beat(), current_clock().time(), time.time()-start)
    wait(1)