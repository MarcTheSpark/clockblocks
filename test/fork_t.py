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


def subsubprocess():
    while True:
        print(current_clock().name, current_clock().beat(), current_clock().time(), time.time()-start)
        wait(0.25, units="time")


def subprocess():
    current_clock().fork(subsubprocess, initial_rate=2, schedule_at=2)
    while True:
        print(current_clock().name, current_clock().beat(), current_clock().time(), time.time()-start)
        wait(1)


c.fork(subprocess, initial_rate=2, schedule_at=1)
while True:
    print(current_clock().name, current_clock().beat(), current_clock().time(), time.time()-start)
    wait(1)