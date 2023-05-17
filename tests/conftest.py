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
from gemseo.disciplines.analytic import AnalyticDiscipline
from matplotlib import pyplot as plt
from numpy import array


@pytest.fixture(scope="package")
def discipline() -> AnalyticDiscipline:
    """The discipline to be calibrated."""
    return AnalyticDiscipline({"y": "a*x", "z": "b*x"})


@pytest.fixture(scope="package")
def reference_data() -> Dataset:
    """The reference data to calibrate the discipline."""
    dataset = Dataset()
    dataset.add_variable("x", array([[0.5], [1.0]]))
    dataset.add_variable("y", array([[1.0], [2.0]]))
    dataset.add_variable("z", array([[-1.0], [-2.0]]))
    return dataset


@pytest.fixture
def baseline_images(request):
    """Return the baseline_images contents.

    Used when the compare_images decorator has indirect set.
    """
    return request.param


@pytest.fixture
def pyplot_close_all():
    """Fixture that prevents figures aggregation with matplotlib pyplot."""
    plt.close("all")
