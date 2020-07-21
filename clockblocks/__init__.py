"""
Clockblocks is a Python library for controlling the flow of musical time, designed with musical applications in mind.
Contents of this package include the `clock` module, which defines the central :class:`~clockblocks.Clock`
class; the `tempo_envelope` module, which defines the :class:`~tempo_envelope.TempoEnvelope` class for mapping
out how a clock's tempo changes with time, as well as the :class:`~tempo_envelope.MetricPhaseTarget` class
for specifying desired metric phases; the `debug` module, which provides utilities for logging in a way that
clarifies which clock logging messages are occurring under; and the `utilities` module, which contains a few
utility functions.
"""

from .clock import Clock, TimeStamp, DeadClockError, ClockKilledError
from .tempo_envelope import TempoEnvelope, MetricPhaseTarget
from .utilities import sleep_precisely, sleep_precisely_until, current_clock, wait, fork, fork_unsynchronized, \
    wait_forever, wait_for_children_to_finish
