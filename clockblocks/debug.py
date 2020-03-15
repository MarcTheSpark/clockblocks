"""
Utilities for logging and debugging processes operating on multiple clocks. (Helps keep track of the order of events.)
"""

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
    # prints out a trace of the last events logged
    print("--- MULTI-THREAD_LOG ---")
    for comment in multi_thread_log[-how_many:]:
        print(comment)
    print("--- END LOG ---")


def clear_multi_log():
    multi_thread_log.clear()


calc_time_starts = {}
calc_time_durations = {}


def log_calc_time_start(tag):
    if tag not in calc_time_starts:
        calc_time_starts[tag] = {}
    calc_time_starts[tag][current_clock()] = time.time()


def log_calc_time_end(tag):
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
