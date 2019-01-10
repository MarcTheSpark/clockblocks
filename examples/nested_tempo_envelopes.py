from clockblocks import Clock
import math


def child_process_1(clock: Clock):
    clock.apply_tempo_function(lambda time: 60 + 30 * math.sin(time), duration_units="time")
    while clock.time() < 40:
        clock.wait(1, units="time")


def child_process_2(clock: Clock):
    clock.apply_tempo_envelope([70, 90, 90, 55, 85, 70], [2, 1.5, 2.5, 1.5, 1.0],
                               curve_shapes=[0, 3, 0, -2, 0], loop=True)
    while clock.time() < 40:
        clock.wait(1, units="time")


master = Clock()
child_1 = master.fork(child_process_1)
child_2 = master.fork(child_process_2)

master.set_rate_target(3, 15)
master.set_rate_target(1, 40, truncate=False)
master.fast_forward_in_time(40)
master.wait_for_children_to_finish()

master.tempo_envelope.show_plot(show_segment_divisions=False)
child_1.tempo_envelope.show_plot(show_segment_divisions=False)
child_2.tempo_envelope.show_plot(show_segment_divisions=False)
child_1.extract_absolute_tempo_envelope().show_plot(show_segment_divisions=False)
child_2.extract_absolute_tempo_envelope().show_plot(show_segment_divisions=False)
