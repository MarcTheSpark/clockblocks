import time
from cb2.clock import Clock
from cb2.utilities import current_clock, wait


c = Clock("MASTER", initial_tempo=30)
start = time.time()


def child():
    current_clock().print_status()
    wait(1)

    # c needs to catch up to the current beat first! Otherwise, this retroactively changes the tempo from the last
    # time it was awake
    c.tempo = 60
    while True:
        current_clock().print_status()
        wait(1)

current_clock().print_status()
wait(1)
c.fork(child, initial_rate=2)
while True:
    current_clock().print_status()
    wait(1)

