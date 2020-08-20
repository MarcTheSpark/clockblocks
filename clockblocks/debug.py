"""
Utilities for logging and debugging processes operating on multiple clocks. (Helps keep track of the order of events.)
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

from .utilities import current_clock
import time

multi_thread_log = []


def log_multi(position: str, *comment_args) -> None:
    """
    Add a log entry with the given position tag.

    :param position: a tag to represent the position in the code where the log is happening
    :param comment_args: arguments alternating between variable names and variable values. For instance, if we call
        :code:`log_multi("position_tag", "x", x, "y", y)` when x is 7 and y is 42, the log entry will read
        "Clock (clock) at checkpoint position_tag: x=7, y=42".
    """
    # useful for keeping track of the order of events between different threads
    comment = ": "
    for i in range(0, len(comment_args), 2):
        if i != 0:
            comment += ", "
        if i+1 < len(comment_args):
            comment += "{}={}".format(comment_args[i], comment_args[i+1])
        else:
            comment += comment_args[i]
    multi_thread_log.append("Clock {} at checkpoint {}{}".format(
        current_clock().name, position, comment if len(comment_args) > 0 else ""
    ))


def print_multi_log(how_many: int) -> None:
    """
    Prints out a trace of the last how_many log statements

    :param how_many: number of statements to print out.
    """
    # prints out a trace of the last events logged
    print("--- MULTI-THREAD_LOG ---")
    for comment in multi_thread_log[-how_many:]:
        print(comment)
    print("--- END LOG ---")


def clear_multi_log() -> None:
    """
    Clears the multi-clock log.
    """
    multi_thread_log.clear()


calc_time_starts = {}
calc_time_durations = {}


def log_calc_time_start(tag: str) -> None:
    """
    Utility to see how long certain stretches of code are taking on different clocks. This call marks the beginning of
    the stretch of code and :function:log_calc_time_end marks the end of the stretch of code. Summaries of how long
    each tagged stretch took are printed at the end of each wait in the master clock.

    :param tag: name used to represent the stretch of code of interest.
    """
    if tag not in calc_time_starts:
        calc_time_starts[tag] = {}
    calc_time_starts[tag][current_clock()] = time.time()


def log_calc_time_end(tag: str) -> None:
    """
    Utility to see how long certain stretches of code are taking on different clocks. This call marks the end of the
    stretch of code and :function:log_calc_time_start marks the beginning of the stretch of code. Summaries of how long
    each tagged stretch took are printed at the end of each wait in the master clock.

    :param tag: name used to represent the stretch of code of interest.
    """
    if tag not in calc_time_starts or current_clock() not in calc_time_starts[tag]:
        raise KeyError("log_calc_time_end called without prior log_calc_time_start.")
    if tag not in calc_time_durations:
        calc_time_durations[tag] = {}
    if current_clock() in calc_time_durations[tag]:
        calc_time_durations[tag][current_clock()] += time.time() - calc_time_starts[tag][current_clock()]
    else:
        calc_time_durations[tag][current_clock()] = time.time() - calc_time_starts[tag][current_clock()]

    del calc_time_starts[tag][current_clock()]


def _print_and_clear_debug_calc_times():
    """
    Prints a summary of how long different tagged stretches of code took to execute on each clock. Also prints tag
    totals and the total time for all tags on all clocks. Used in combination with :function:log_calc_time_start and
    :function:log_calc_time_end, which are used to mark the start and end point of each tag.

    Note that this is called automatically by the clock when log_calc_time_start and log_calc_time_end are being used.
    """
    global calc_time_starts, calc_time_durations
    if len(calc_time_durations) == 0:
        calc_time_starts.clear()
        return False  # return of False means that no tags were tracked, so nothing to print
    print("-------- CALC TIMES ---------")
    total_for_all_tags = 0
    for tag in calc_time_durations:
        print("  {}:".format(tag))
        total = 0
        for clock in calc_time_durations[tag]:
            print("    {}: {}".format(clock.name, calc_time_durations[tag][clock]))
            total += calc_time_durations[tag][clock]
        total_for_all_tags += total
        print("  Total for tag {}: {}".format(tag, total))
    print("Total for all tags: {}".format(total_for_all_tags))

    calc_time_starts.clear()
    calc_time_durations.clear()
    return True  # return of True means that something was printed here
