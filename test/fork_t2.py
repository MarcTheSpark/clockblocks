# from cb2.clock import Clock
# from cb2.tempo_envelope import TempoHistory
# import math
#
# th = TempoHistory(90)
# th.apply_function(lambda b: 20 * math.sin(b) + 60, duration_units="time")
# # th.show_plot()
# # th.set_tempo_target(60, 10)
# th.advance(6.1, "time")
# # print(th.beat_at_time(9))
# # th.set_tempo_target(60, 10)
# th.show_plot()
# exit()

import threading
import time
from cb2.clock import Clock
from cb2.utilities import current_clock, wait
from cb2.scheduler import get_scheduler
from cb2.tempo_envelope import TempoEnvelope
import math
import logging


c = Clock("MASTER", initial_tempo=30)
start = time.time()


def subprocess():
    print(current_clock().name, current_clock().beat(), current_clock().time(), time.time() - start)
    # print(current_clock().name, current_clock().beat(), current_clock().time(), time.time() - start)
    wait(1)

    # c needs to catch up to the current beat first! Otherwise, this retroactively changes the tempo from the last
    # time it was awake
    c.tempo = 60
    while True:
        print(current_clock().name, current_clock().beat(), current_clock().time(), time.time()-start)
        wait(1)

print(current_clock().name, current_clock().beat(), current_clock().time(), time.time()-start)
wait(1)
c.fork(subprocess, initial_rate=2)
while True:
    print(current_clock().name, current_clock().beat(), current_clock().time(), time.time()-start)
    wait(1)