"""
Exception hierarchy for clockblocks. Kept in a dependency-free module of its own so that any module
(``clock``, ``utilities``, ``scheduler``, …) can import these at the top level without risking an
import cycle — several of them reference each other's errors but none of them should have to defer the
import to function-call time.
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


class ClockblocksError(Exception):
    """Base class for clockblocks errors."""
    pass


class ClockKilledError(ClockblocksError):
    """Raised inside a forked clock's own thread when its wait is woken by kill(),
    so the fork wrapper can unwind the user function cleanly."""
    pass


class DeadClockError(ClockblocksError):
    """Raised when something tries to wait or fork on a clock that's no longer ALIVE
    (either killed already, or — for fork — still PENDING in its start_delay).

    Note that, when a master clock is killed, context-inferring operations like `wait` and `fork` will raise a
    NoActiveClockError, whereas direct calls of `c.wait` or `c.fork` on a reference to the dead clock will raise
    a DeadClockError."""
    pass


class WrongThreadError(ClockblocksError):
    """Raised when wait() is called from a thread that doesn't own the clock.
    Each clock has exactly one owning thread (the one running its forked function,
    or the main thread for the master); calling wait() from any other thread would
    block the wrong thread and corrupt the clock's bookkeeping."""
    pass


class NoActiveClockError(ClockblocksError):
    """Raised when a context-inferring clock operation (the module-level wait/fork/etc.) is attempted from
    a thread that has no clock active on it. Establish one with fork() / run_as_server(), or call the method
    on a Clock object directly.

    Note that, when a master clock is killed, context-inferring operations like `wait` and `fork` will raise a
    NoActiveClockError, whereas direct calls of `c.wait` or `c.fork` on a reference to the dead clock will raise
    a DeadClockError.
    """
    pass


class NotMasterClockError(ClockblocksError):
    """Raised by operations that are only valid on the master (top-level) clock — e.g.
    run_as_server() — when called on a child clock."""
    pass


class SchedulerHeldError(ClockblocksError):
    """Raised when wait() is called from a thread that is holding the scheduler via hold_scheduler()
    — most commonly inside a MIDI / OSC / HID callback, which runs under such a hold. Trying to pass
    time there leads to deadlock: the wait parks, waiting for the run loop to fire its wake-up, but the
    run loop can't proceed until the hold is released. Schedule the timed work with fork() instead."""
    pass
