"""
Clockblocks is a Python library for controlling the flow of musical time, designed with musical applications in mind.
Contents of this package include the `clock` module, which defines the central :class:`clockblocks.Clock`
class; the `tempo_envelope` module, which defines the :class:`tempo_envelope.TempoEnvelope` class for mapping
out how a :class:`Clock`'s tempo changes with time as well as the :class:`tempo_envelope.MetricPhaseTarget class
for specifying desired metric phases; the `debug` module, which provides utilities for keeping
logging in a way that clarifies which clock logging messages are occurring under; and the `utilities` module,
which contains a few utility functions.
"""

from .clock import *
