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
"""Test the calibration metric MeanError."""

from __future__ import annotations

import pytest
from numpy import array
from numpy import nan
from numpy import ndarray
from numpy import ones
from numpy.testing import assert_array_equal

from gemseo_calibration.metrics.iae import IAE
from gemseo_calibration.metrics.mae import MAE


@pytest.fixture(scope="module")
def reference_data():
    """Synthetic reference data containing two observations."""
    return {
        "y": array([[1.0], [2.0], [2.0], [nan]]),
        "z": array([[1.0] * 3, [2.0] * 3, [2.0] * 3, [nan, 2.0, 2.0]]),
    }


@pytest.fixture(scope="module")
def model_data() -> dict[str, ndarray]:
    """Synthetic model data corresponding to the input values of the reference data."""
    return {
        "x": array([[1.0], [2.0], [2.0], [2.0]]),
        "y": array([[3.0], [4.0], [nan], [4.0]]),
        "z": array([
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [nan, 4.0, 5.0],
            [3.0, 4.0, 5.0],
        ]),
    }


@pytest.mark.parametrize(("output_name", "expected"), [("y", 2.0), ("z", 2.2)])
def test_mean_error(reference_data, model_data, output_name, expected):
    """Test that the mean error works correctly for different outputs."""
    mae = MAE(output_name)
    mae.set_reference_data(reference_data)
    assert mae.func(model_data) == expected


@pytest.mark.parametrize(
    ("output_name", "mesh_name", "expected"),
    [("y", None, 2.0), ("y", "m", 6.5)],
)
def test_mean_error_with_mesh(output_name, mesh_name, expected):
    """Test that the mean error works correctly in presence of indexed variables."""
    reference_data = {"y": array([[1.0, 1.0, 1.0]]), "m": array([[0.0, 1.0, 3.0]])}
    model_data = {"y": array([[2.0, 3.0, 4.0]]), "m": array([[0.0, 1.0, 3.0]])}
    if mesh_name is None:
        metric = MAE(output_name)
    else:
        metric = IAE(output_name, mesh_name)
        assert metric.full_output_name == "y[m]"
    metric.set_reference_data(reference_data)
    assert metric.func(model_data) == expected


@pytest.mark.parametrize(
    ("reference_mesh", "expected_metric"),
    [([0.0, 1.0, 2.0, 3.0], 6.5), ([0.0, 3.0], 6.0)],
)
def test_mean_error_with_interpolation_over_reference_mesh(
    reference_mesh, expected_metric
):
    """Test that integrated metrics handle interpolation over reference mesh."""
    reference_data = {"y": ones((1, len(reference_mesh))), "m": array([reference_mesh])}
    model_data = {"y": array([[2.0, 3.0, 4.0]]), "m": array([[0.0, 1.0, 3.0]])}
    metric = IAE("y", "m")
    metric.set_reference_data(reference_data)
    assert metric.func(model_data) == expected_metric
    assert_array_equal(metric.reference_mesh, reference_data["m"])
