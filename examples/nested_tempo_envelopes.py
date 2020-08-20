#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  #
#  This file is part of SCAMP (Suite for Computer-Assisted Music in Python)                      #
#  Copyright © 2020 Marc Evanstein <marc@marcevanstein.com>.                                     #
#                                                                                                #
#  This program is free software: you can redistribute it and/or modify it under the terms of    #
#  the GNU General Public License as published by the Free Software Foundation, either version   #
#  3 of the License, or (at your option) any later version.                                      #
#                                                                                                #
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;     #
#  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.     #
#  See the GNU General Public License for more details.                                          #
#                                                                                                #
#  You should have received a copy of the GNU General Public License along with this program.    #
#  If not, see <http://www.gnu.org/licenses/>.                                                   #
#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  #

from clockblocks import Clock
import math


def child_process_1(clock: Clock):
    clock.apply_tempo_function(lambda time: 60 + 30 * math.sin(time), duration_units="time")
    while clock.time() < 40:
        clock.wait(1, units="time")


def child_process_2(clock: Clock):
    clock.tempo = 70
    clock.set_tempo_targets([90, 90, 55, 85, 70], [2, 3.5, 6, 7.5, 8.0], curve_shapes=[0, 3, 0, -2, 0], loop=True)
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
