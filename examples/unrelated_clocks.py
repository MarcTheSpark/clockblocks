"""
Two unrelated (master) clocks running at different tempi.
"""

import threading

from clockblocks.clock import Clock
import math

c = Clock(initial_tempo=90)
# c.tempo_history.apply_function(lambda b: 120 + math.sin(b/10) * 80, duration_units="time")

def other_clock():
    c2 = Clock()
    while True:
        print("c2", c2.beat(), c2.time(), c2.tempo)
        c2.wait(1)

threading.Thread(target=other_clock, daemon=True).start()

while True:
    print("c1", c.beat(), c.time(), c.tempo)
    c.wait(1)

