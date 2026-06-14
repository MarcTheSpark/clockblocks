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

from cb2.clock import Clock, ClockState, ClockFamilyOptions
from cb2.exceptions import ClockblocksError, ClockKilledError, DeadClockError, WrongThreadError, \
    NoActiveClockError, NotMasterClockError
from cb2.tempo_envelope import TempoEnvelope, TempoHistory
from cb2.metric_phase import MetricPhaseTarget
from cb2.moment import Moment, ResolvableMoment
from cb2.time_stamp import TimeStamp
from cb2.utilities import current_clock, wait, wait_forever, wait_for_children_to_finish, fork, fork_unsynchronized, \
    set_tempo, set_rate, set_beat_length, get_tempo, get_rate, get_beat_length, \
    set_tempo_target, set_rate_target, set_beat_length_target, \
    set_tempo_targets, set_rate_targets, set_beat_length_targets, \
    apply_tempo_function, apply_rate_function, apply_beat_length_function, \
    apply_tempo_envelope, stop_tempo_loop_or_function
from cb2.enums import DurationUnits, TempoUnits

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    pass
