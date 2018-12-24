"""
A simple example in which the master clock forks two parallel (sister) processes and everything moves at the same tempo.
(Although sister 2 prints its beat more frequently than sister 1).

Note that sister1 ends after 5 beats, whereas sister2 ends after 10 beats.

This is similar conceptually to multi-threading, except that the master clock handles all the actual waiting, and
thereby ensures that all its children remain perfectly synchronized.
"""

from clockblocks import Clock

master = Clock()


def sister1(my_clock):
    # sister 1 prints once per second for the first five seconds
    while my_clock.beats() < 6:
        print("Clock 1 at beat {}".format(round(my_clock.beats(), 3)))
        my_clock.wait(1)


def sister2(my_clock):
    # sister 1 prints thrice per second for the first 10 seconds
    while my_clock.beats() < 3:
        print("Clock 2 at beat {}".
              format(round(my_clock.beats(), 3)))
        my_clock.wait(1/3)


master.fork(sister1)
master.fork(sister2)
master.wait_for_children_to_finish()
