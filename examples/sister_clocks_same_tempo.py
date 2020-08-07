"""
A simple example in which the master clock forks two parallel (sister) processes and everything moves at the same tempo.
(Although sister 2 prints its beat more frequently than sister 1).

Note that sister1 ends after 5 beats, whereas sister2 ends after 10 beats.

This is similar conceptually to multi-threading, except that the master clock handles all the actual waiting, and
thereby ensures that all its children remain perfectly synchronized.
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

from clockblocks import Clock

master = Clock()


def sister1(my_clock):
    # sister 1 prints once per second for the first five seconds
    while my_clock.beat() < 6:
        print("Clock 1 at beat {}".format(round(my_clock.beat(), 3)))
        my_clock.wait(1)


def sister2(my_clock):
    # sister 1 prints thrice per second for the first 10 seconds
    while my_clock.beat() < 3:
        print("Clock 2 at beat {}".
              format(round(my_clock.beat(), 3)))
        my_clock.wait(1/3)


master.fork(sister1)
master.fork(sister2)
master.wait_for_children_to_finish()
