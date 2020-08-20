"""
Demonstration of multiple generations of nested clocks running at different tempi.

When running nested clocks, it is important to note that a tempo of 60 is the same as a rate of 1, meaning that the
clock runs at the same speed as its parent. A tempo of 120  (or rate of 120 / 60 = 2) means that it runs at double the
speed of its parent, and a tempo of 30 (or rate of 1/2) means it runs at half the speed of its parent. All clocks
default to a tempo of 60 / rate of 1 unless otherwise indicated.

In this script, the master clock counts beats and forks a child clock once every 8 beats.

That child clock runs at twice the tempo of the master clock, since it is given an "initial_tempo" of 120 in the call
to "fork". It counts four beats at that tempo and then forks the grandchild process. At that point, it's done printing
and waits for its child (the grandchild) to finish. (Note that if it didn't call "wait_for_children_to_finish" it would
end abruptly, causing the grandchild process to terminate before it even gets going.)

The grandchild process starts by setting its tempo to 120. This means that it is going twice as fast as its parent (the
child), which in turn was going twice as fast as the master clock, so the grandchild starts out at 4 times the speed of
the master clock. However, it slows down to a target tempo of 40 over the course of its 8 beats of life, and that final
tempo represents a rate of (40 / 60) * (120 / 60) = 4/3 the speed of the master clock.

The final thing to note is that the master clock speeds up to a tempo of 180 (from an implicit initial tempo of 60) over
the course of 40 beats. This accelerates all of its descendants as well.
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

from clockblocks import *


def child_process(my_clock: Clock):
    while my_clock.beat() < 4:
        print("{} at beat {}".format(my_clock.name, my_clock.beat()))
        wait(1)
    my_clock.fork(grandchild_process, name="GRANDCHILD")
    my_clock.wait_for_children_to_finish()


def grandchild_process(my_clock: Clock):
    my_clock.tempo = 120
    my_clock.set_tempo_target(40, 8)
    while my_clock.beat() < 8:
        print("{} at beat {}".format(my_clock.name, my_clock.beat()))
        my_clock.wait(1)


master = Clock("MASTER")

master.set_tempo_target(180, 40)
while master.beat() < 40:
    print("{} at beat {}".format(master.name, master.beat()))
    if master.beat() % 8 == 0:
        master.fork(child_process, name="CHILD", initial_tempo=120)
    wait(1)
