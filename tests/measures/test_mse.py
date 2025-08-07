# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""Test the calibration metrics MSE and ISE."""

from __future__ import annotations

import pytest
from numpy import array
from numpy.testing import assert_array_equal

from gemseo_calibration.metrics.ise import ISE
from gemseo_calibration.metrics.mse import MSE


@pytest.mark.parametrize("metric", [MSE, ISE])
def test_compute_output_error(metric):
    """Test that the static method _compute_output_error returns a squared error."""
    output_error = metric._compare_data(array([0.0]), array([2.0]))
    assert_array_equal(output_error, array([4.0]))
