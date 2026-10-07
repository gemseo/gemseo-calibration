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
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Test the class DataVersusModel."""

from __future__ import annotations

import pytest
from gemseo.discipline import AnalyticDiscipline
from gemseo.optimization import NLOPT_COBYLA_Settings
from gemseo.space import DesignSpace
from numpy import array

from gemseo_calibration.metrics.settings import CalibrationMetricSettings
from gemseo_calibration.post.data_versus_model.post import DataVersusModel
from gemseo_calibration.post.data_versus_model.settings import DataVersusModel_Settings
from gemseo_calibration.scenario import CalibrationScenario


@pytest.fixture(scope="module")
def calibration_scenario() -> CalibrationScenario:
    """A calibration scenario."""
    model = AnalyticDiscipline({"y": "a*x", "z": "b*x"}, name="model")
    reference = AnalyticDiscipline({"y": "2*x", "z": "3*x"}, name="reference")

    prior = DesignSpace()
    prior.add_variable("a", lower_bound=0.0, upper_bound=10.0, value=0.0)
    prior.add_variable("b", lower_bound=0.0, upper_bound=10.0, value=0.0)

    reference.set_cache("MemoryFullCache")
    reference.execute({"x": array([1.0])})
    reference.execute({"x": array([2.0])})
    reference_data = reference.cache.to_dataset().to_dict_of_arrays(False)

    calibration = CalibrationScenario(
        model,
        "x",
        [
            CalibrationMetricSettings(output_name="y", metric_name="MSE"),
            CalibrationMetricSettings(output_name="z", metric_name="MSE"),
        ],
        prior,
    )
    calibration.execute(
        NLOPT_COBYLA_Settings(max_iter=10), reference_data=reference_data
    )
    return calibration


@pytest.mark.parametrize("output", ["y", "z"])
def test_plot(output, calibration_scenario, post_factory, snapshot_matplotlib):
    """Test images created by DataVersusModel._plot against references."""
    post_factory.create(
        "DataVersusModel",
        calibration_scenario.formulation.problem,
        calibration_scenario.calibrator.reference_data,
        calibration_scenario.prior_model_data,
        calibration_scenario.posterior_model_data,
    ).execute(DataVersusModel_Settings(output=output, save=False))


def test_factory_plot(post_factory, calibration_scenario):
    """Check CalibrationPostFactory.execute()."""
    post = post_factory.execute(
        calibration_scenario.formulation.problem,
        calibration_scenario.calibrator.reference_data,
        calibration_scenario.prior_model_data,
        calibration_scenario.posterior_model_data,
        DataVersusModel_Settings(output="y", save=False),
    )
    assert isinstance(post, DataVersusModel)
