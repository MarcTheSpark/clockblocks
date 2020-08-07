"""
Like sister_clocks_same_tempo.py, except that sister 2 is running at a tempo of 180, making it 3 times faster than
sister 1. As a result, it goes through beats more quickly.

Note that sister2 ends when its _time_ reaches 10. This is the same thing as saying when its parent reaches beat 10.
The time of a clock runs the same regardless of its rate, and it represents the inherited rate from its parent, and
from its parent's parent, etc.
"""

#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  #
#  SCAMP (Suite for Computer-Assisted Music in Python)                                           #
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

from clockblocks import *

master = Clock()


def sister1(my_clock):
    # sister 1 prints once per second for the first five seconds
    while my_clock.beat() < 5:
        print("Sister 1 at beat {}".format(my_clock.beat()))
        wait(1)


def sister2(my_clock):
    # sister 1 prints thrice per second for the first 10 seconds
    while my_clock.time() < 10:
        print("Sister 2 at beat {}".format(my_clock.beat()))
        wait(1)


master.fork(sister1)
master.fork(sister2, initial_tempo=180)
master.wait_for_children_to_finish()
