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

import pickle
import unittest
import warnings

from clockblocks.clock import Clock

POSITION_PROPERTIES = ("beat", "time", "absolute_rate", "absolute_tempo", "absolute_beat_length")


class PositionPropertyTestCase(unittest.TestCase):
    """
    Tests for the 1.1 property spelling of the position accessors (beat, time, absolute_*), and for the
    transitional _CallableFloat shim that keeps the pre-1.1 method spelling working with a DeprecationWarning.
    """

    def setUp(self):
        self.master = Clock(name="master")

    def tearDown(self):
        self.master.kill()

    def test_position_properties_are_plain_float_values(self):
        for name in POSITION_PROPERTIES:
            value = getattr(self.master, name)
            self.assertIsInstance(value, float, name)
        self.assertEqual(self.master.beat, 0.0)
        self.assertEqual(self.master.time, 0.0)
        self.assertEqual(self.master.absolute_rate, 1.0)
        self.assertEqual(self.master.absolute_tempo, 60.0)
        self.assertEqual(self.master.absolute_beat_length, 1.0)

    def test_property_advances_with_wait(self):
        self.master.wait(0.5)
        self.assertAlmostEqual(self.master.beat, 0.5)
        self.assertAlmostEqual(self.master.time, 0.5)

    def test_position_properties_are_read_only(self):
        for name in POSITION_PROPERTIES:
            with self.assertRaises(AttributeError, msg=name):
                setattr(self.master, name, 5)

    def test_legacy_call_spelling_warns_and_returns_same_value(self):
        self.master.wait(0.25)
        for name in POSITION_PROPERTIES:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                called_value = getattr(self.master, name)()
            self.assertEqual(len(caught), 1, name)
            self.assertTrue(issubclass(caught[0].category, DeprecationWarning), name)
            self.assertIn(name, str(caught[0].message))
            self.assertEqual(called_value, getattr(self.master, name), name)
            self.assertIs(type(called_value), float, name)

    def test_shim_value_behaves_as_float(self):
        self.master.wait(0.5)
        beat = self.master.beat
        # arithmetic degrades to plain float; comparison, hashing, and pickling behave like a float
        self.assertIs(type(beat + 1), float)
        self.assertEqual(beat * 2, 1.0)
        self.assertEqual(hash(beat), hash(0.5))
        self.assertEqual(pickle.loads(pickle.dumps(beat)), 0.5)

    def test_projected_accessors_remain_methods(self):
        # projected_* are wall-clock samples, deliberately kept as methods
        self.assertTrue(callable(Clock.projected_beat))
        self.assertIsInstance(self.master.projected_beat(), float)
        self.assertIsInstance(self.master.projected_time(), float)


if __name__ == '__main__':
    unittest.main()
