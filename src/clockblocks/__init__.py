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
"""
Clockblocks is a library for controlling the flow of musical time, part of SCAMP (Suite for Computer-Assisted
Music in Python).

Each :class:`~clockblocks.clock.Clock` runs on its own thread, and clocks form families under a single master
clock, whose :class:`~clockblocks.scheduler.Scheduler` wakes each thread back up when its call to
:func:`~clockblocks.utilities.wait` is due. The scheduler keeps the whole clock family coordinated under a
shared, ideal timeline, which is not polluted by the small delays that user code inevitably incurs. How
aggressively a clock catches up when it does fall behind is set by its timing policy, which ranges from
waiting out each delay in full to staying pinned to the absolute schedule.

The rate at which beats pass within a clock is given by its tempo, which may vary over time, as described by
a :class:`~clockblocks.tempo_envelope.TempoEnvelope`. Clocks are also nestable: :func:`~clockblocks.utilities.fork`
spawns a child clock running a function in a parallel, subordinate timeline. A child clock's tempo is felt
relative to that of its parent, so the true rate at which time passes in a clock is the product of the clock's
own rate and that of each of its ancestors.

Taken together, these features allow a composer to create multiple, polyphonic (and perhaps poly-tempo) layers
of music, each with its own tempo curve, all coordinated under a master clock.
"""

from importlib.metadata import version, PackageNotFoundError

from clockblocks.clock import Clock, ClockState, ClockFamilyOptions
from clockblocks.scheduler import TimingBackend, CompressedTime
from clockblocks.exceptions import ClockblocksError, ClockKilledError, DeadClockError, WrongThreadError, \
    NoActiveClockError, NotMasterClockError
from clockblocks.tempo_envelope import TempoEnvelope, TempoHistory
from clockblocks.metric_phase import MetricPhaseTarget
from clockblocks.moment import Moment, ResolvableMoment
from clockblocks.time_stamp import TimeStamp, TimeStampInterval
from clockblocks.utilities import current_clock, wait, wait_until, wait_forever, wait_for_children_to_finish, fork, \
    set_tempo, set_rate, set_beat_length, get_tempo, get_rate, get_beat_length, get_beat, get_time, \
    set_tempo_target, set_rate_target, set_beat_length_target, \
    set_tempo_targets, set_rate_targets, set_beat_length_targets, \
    apply_tempo_function, apply_rate_function, apply_beat_length_function, \
    apply_tempo_envelope, stop_tempo_loop_or_function
from clockblocks.enums import DurationUnits, TempoUnits

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    pass
