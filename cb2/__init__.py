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

from importlib.metadata import version, PackageNotFoundError

from cb2.clock import Clock, ClockKilledError, DeadClockError, ClockblocksError, WrongThreadError, \
    NoActiveClockError, NotMasterClockError, ClockState, ClockFamilyOptions
from cb2.tempo_envelope import TempoEnvelope, TempoHistory
from cb2.metric_phase import MetricPhaseTarget
from cb2.moment import Moment, ResolvableMoment
from cb2.time_stamp import TimeStamp
from cb2.utilities import current_clock, wait, wait_forever, wait_for_children_to_finish, fork, fork_unsynchronized
from cb2.enums import DurationUnits, TempoUnits

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    pass
