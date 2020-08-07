"""
Clockblocks is a Python library for controlling the flow of musical time, designed with musical applications in mind.
Contents of this package include the `clock` module, which defines the central :class:`~clockblocks.Clock`
class; the `tempo_envelope` module, which defines the :class:`~tempo_envelope.TempoEnvelope` class for mapping
out how a clock's tempo changes with time, as well as the :class:`~tempo_envelope.MetricPhaseTarget` class
for specifying desired metric phases; the `debug` module, which provides utilities for logging in a way that
clarifies which clock logging messages are occurring under; and the `utilities` module, which contains a few
utility functions.
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

from .clock import Clock, TimeStamp, DeadClockError, ClockKilledError
from .tempo_envelope import TempoEnvelope, MetricPhaseTarget
from .utilities import sleep_precisely, sleep_precisely_until, current_clock, wait, fork, fork_unsynchronized, \
    wait_forever, wait_for_children_to_finish
