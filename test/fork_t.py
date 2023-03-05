import threading
import time

from cb2.clock import Clock
from cb2.utilities import wait, current_clock
from cb2.tempo_envelope import TempoEnvelope
import math
import logging

logging.basicConfig(level=logging.DEBUG)

c = Clock(initial_tempo=60)


def subprocess():
    while True:
        logging.debug('\x1b[38;5;39m'+ f"sub: {current_clock().beat(), current_clock().time(), current_clock().scheduler.current_stage}" + "\x1b[0m")
        wait(1)
c.fork(subprocess, initial_rate=2)
while True:

    logging.debug('\x1b[38;5;39m' + f"TOPCLOCK: {c.beat(), c.time(), c.scheduler.current_stage}" + "\x1b[0m")
    wait(1)