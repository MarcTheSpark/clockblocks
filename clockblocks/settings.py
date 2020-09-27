"""
Global preferences and settings for clockblocks.
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

#: Throw a warning when the master clock is running more than this many seconds behind on a longer wait call
running_behind_warning_threshold_long = 0.010
#: Throw a warning when the master clock is running more than this many seconds behind on a short/zero-length wait call
running_behind_warning_threshold_short = 0.025
#: Throw a warning when catching up child clocks takes longer than this, and master clock is running behind
catching_up_child_clocks_threshold_min = 0.001
#: Throw a warning when catching up child clocks takes longer than this, but master clock is not running behind
catching_up_child_clocks_threshold_max = 0.005
