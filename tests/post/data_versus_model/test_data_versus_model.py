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
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.utils.testing.helpers import image_comparison
from numpy import array

from gemseo_calibration.metrics.settings import CalibrationMetricSettings
from gemseo_calibration.post.data_versus_model.post import DataVersusModel
from gemseo_calibration.scenario import CalibrationScenario


@pytest.fixture(scope="module")
def calibration_scenario() -> CalibrationScenario:
    """A calibration scenario."""
    model = AnalyticDiscipline({"y": "a*x", "z": "b*x"}, name="model")
    reference = AnalyticDiscipline({"y": "2*x", "z": "3*x"}, name="reference")

    prior = ParameterSpace()
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
        algo_name="NLOPT_COBYLA", reference_data=reference_data, max_iter=10
    )
    return calibration


# the test parameters, it maps a test name to the inputs and references outputs:
# - the kwargs to be passed to DataVersusModel._plot
# - the expected file names without extension to be compared
TEST_PARAMETERS = {
    "output_y": ({"output": "y"}, ["output_y"]),
    "output_z": ({"output": "z"}, ["output_z"]),
}


@pytest.mark.parametrize(
    ("kwargs", "baseline_images"),
    TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=TEST_PARAMETERS.keys(),
)
@image_comparison(None)
def test_plot(
    kwargs, baseline_images, calibration_scenario, pyplot_close_all, post_factory
):
    """Test images created by DataVersusModel._plot against references."""
    post_factory.create(
        "DataVersusModel",
        calibration_scenario.formulation.optimization_problem,
        calibration_scenario.calibrator.reference_data,
        calibration_scenario.prior_model_data,
        calibration_scenario.posterior_model_data,
    ).execute(save=False, **kwargs)


def test_factory_plot(post_factory, calibration_scenario):
    """Check CalibrationPostFactory.execute()."""
    post = post_factory.execute(
        calibration_scenario.formulation.optimization_problem,
        calibration_scenario.calibrator.reference_data,
        calibration_scenario.prior_model_data,
        calibration_scenario.posterior_model_data,
        "DataVersusModel",
        output="y",
        save=False,
    )
    assert isinstance(post, DataVersusModel)
