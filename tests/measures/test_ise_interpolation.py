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
from __future__ import annotations

import pytest
from gemseo.datasets.dataset import Dataset
from numpy import all as np_all
from numpy import diff
from numpy import full
from numpy import hstack
from numpy import linspace
from numpy import ndarray

from gemseo_calibration.metrics.ise import ISE


def create_dataset(
    mesh_size: int,
    y: ndarray | None = None,
    mesh_start: float = 0.0,
    mesh_end: float = 1.0,
    is_decreasing_mesh: bool = False,
):
    """Generate a dataset containing two vector variables of same length.

    The abscissa can have monotonically increasing or decreasing values.
    The default y values are constant and equal to 1.
    """
    mesh = linspace(mesh_start, mesh_end, mesh_size)
    mesh_is_decreasing = np_all(diff(mesh) < 0)
    if mesh_is_decreasing != is_decreasing_mesh:
        mesh = mesh[::-1]
    y = y if y is not None else full((mesh_size), 1.0)
    return Dataset.from_array(
        data=[hstack([y, mesh])],
        variable_names=["y", "y_mesh"],
        variable_names_to_n_components={"y": mesh_size, "y_mesh": mesh_size},
    )


@pytest.mark.parametrize("is_decreasing_ref_mesh", [False, True])
@pytest.mark.parametrize("is_decreasing_model_mesh", [False, True])
def test_interpolation(is_decreasing_ref_mesh, is_decreasing_model_mesh):
    """Check that the interpolation of reference data on model data is correct
    whether the reference or model abscissa had monotically increasing or decreasing
    values."""

    mesh_size = 5
    reference_dataset = create_dataset(
        mesh_size,
        is_decreasing_mesh=is_decreasing_ref_mesh,
        y=linspace(1.0, 2.0, mesh_size),
    )

    mesh_size = 10
    model_dataset = create_dataset(
        mesh_size,
        is_decreasing_mesh=is_decreasing_model_mesh,
        mesh_start=0.0,
        mesh_end=1.0,
        y=linspace(3.0, 4.0, mesh_size),
    )

    metric = ISE("y", "y_mesh")
    metric.set_reference_data(reference_dataset.to_dict_of_arrays(False))
    result = metric._evaluate_metric(model_dataset.to_dict_of_arrays(False))
    assert result == pytest.approx(4.0, rel=0.1)
