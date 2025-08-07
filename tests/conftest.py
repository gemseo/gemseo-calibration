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

from pathlib import Path

import pytest
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.utils.testing.pytest_conftest import *  # noqa: F401, F403
from matplotlib import pyplot as plt
from numpy import array
from numpy import ndarray

from gemseo_calibration.metrics.factory import CalibrationMetricFactory
from gemseo_calibration.post.factory import CalibrationPostFactory

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="package")
def discipline() -> AnalyticDiscipline:
    """The discipline to be calibrated."""
    return AnalyticDiscipline({"y": "a*x", "z": "b*x"})


@pytest.fixture(scope="package")
def reference_data() -> dict[str, ndarray]:
    """The reference data to calibrate the discipline."""
    return {
        "x": array([[0.5], [1.0]]),
        "y": array([[1.0], [2.0]]),
        "z": array([[-1.0], [-2.0]]),
    }


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


@pytest.fixture
def metric_factory(reset_factory, monkeypatch) -> CalibrationMetricFactory:
    monkeypatch.setenv("GEMSEO_PATH", DATA)
    return CalibrationMetricFactory()


@pytest.fixture
def post_factory(reset_factory, monkeypatch) -> CalibrationPostFactory:
    """The factory of post-processors dedicated to calibration."""
    monkeypatch.setenv("GEMSEO_PATH", DATA)
    return CalibrationPostFactory()
