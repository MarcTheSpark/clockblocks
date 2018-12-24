from clockblocks import *
import math


master = Clock()

master.set_rate_target(2, 40)


def child_process():
    while True:
        wait(1)


child = master.fork(child_process)

sin_env = TempoEnvelope.from_function(lambda x: 1 + math.sin(x) / 4, 0, 100, units="rate")
child.apply_tempo_envelope(TempoEnvelope.from_function(lambda x: 1 + math.sin(x) / 4, 0, 40, units="rate"))

master.fast_forward_in_beats(10)
master.wait(10)

abs_tempo_env = child.extract_absolute_tempo_envelope(step_size=0.5)

abs_tempo_env.show_plot(show_segment_divisions=False)
