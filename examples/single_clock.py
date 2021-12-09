"""
Demo of a single clock changing tempo and the resulting beat length graph.
"""

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

clock = Clock(initial_tempo=60)


def log_timing():
    print("Beat: {}, Time: {}, Current Tempo: {}".format(clock.beat(), clock.time(), round(clock.tempo, 2)))


while clock.beat() < 4:
    log_timing()
    clock.wait(1)

# sudden change to 120 bpm (i.e. 2 beats per second)
clock.tempo = 120

while clock.beat() < 8:
    log_timing()
    clock.wait(1)

# gradually slow to 30 bpm over the course of 8 beats
clock.set_tempo_target(30, 8)

while clock.beat() < 20:
    log_timing()
    clock.wait(1)

clock.tempo_history.show_plot(title="Example TempoHistory", units="beatlength", y_range=(0, 2.5), resolution=200)
