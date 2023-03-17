import threading
import time

from cb2.clock import Clock
from cb2.utilities import wait, current_clock
from cb2.scheduler import get_scheduler
from cb2.tempo_envelope import TempoEnvelope
import math
import logging


c = Clock("MASTER", initial_tempo=20)

def subsubprocess():
    while True:
        print(current_clock().name, current_clock().beat(), current_clock().time(), c.wall_time_in_scheduler())
        wait(0.5)

def subprocess():
    while True:
        if current_clock().beat() == 2:
            current_clock().fork(subsubprocess, initial_rate=2)
        print(current_clock().name, current_clock().beat(), current_clock().time(), c.wall_time_in_scheduler())
        wait(1)


c.fork(subprocess, initial_rate=2)
while True:
    print(current_clock().name, current_clock().beat(), current_clock().time(), c.wall_time_in_scheduler())
    wait(1)