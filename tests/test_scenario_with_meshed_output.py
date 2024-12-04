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

from typing import TYPE_CHECKING

import pytest
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.discipline.discipline import Discipline
from numpy import array
from numpy import linspace
from numpy import ndarray

from gemseo_calibration.metrics.settings import CalibrationMetricSettings
from gemseo_calibration.scenario import CalibrationScenario

if TYPE_CHECKING:
    from gemseo.typing import StrKeyMapping


class Model(Discipline):
    """The model to be calibrated."""

    def __init__(self):  # noqa: D107
        super().__init__()
        self.input_grammar.update_from_names(["x", "a", "b"])
        self.output_grammar.update_from_names(["y", "z", "mesh"])
        self.default_input_data = {
            "x": array([0.0]),
            "a": array([0.0]),
            "b": array([0.0]),
        }

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        x_input = self.io.data["x"]
        a_parameter = self.io.data["a"]
        b_parameter = self.io.data["b"]
        z_mesh = linspace(0, 1, 5)
        y_output = a_parameter * x_input * z_mesh
        z_output = b_parameter * x_input[0] * z_mesh
        self.io.update_output_data({"y": y_output, "z": z_output, "mesh": z_mesh})


class ReferenceModel(Discipline):
    """The model to be approximated."""

    def __init__(self):  # noqa: D107
        super().__init__()
        self.input_grammar.update_from_names(["x"])
        self.output_grammar.update_from_names(["y", "z", "mesh"])
        self.default_input_data = {"x": array([0.0])}

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        x_input = self.io.data["x"]
        z_mesh = linspace(0, 1, 5)
        y_output = 2 * x_input * z_mesh
        z_output = 3 * x_input[0] * z_mesh
        self.io.update_output_data({"y": y_output, "z": z_output, "mesh": z_mesh})


@pytest.fixture(scope="module")
def calibration_space() -> ParameterSpace:
    """The calibration space."""
    space = ParameterSpace()
    space.add_variable("a", lower_bound=0.0, upper_bound=10.0, value=0.0)
    space.add_variable("b", lower_bound=0.0, upper_bound=10.0, value=0.0)
    return space


@pytest.fixture(scope="module")
def reference_data() -> dict[str, ndarray]:
    """The reference dataset."""
    reference = ReferenceModel()
    reference.set_cache("MemoryFullCache")
    reference.execute({"x": array([1.0])})
    reference.execute({"x": array([2.0])})
    return reference.cache.to_dataset().to_dict_of_arrays(False)


def test_execute(reference_data, calibration_space):
    """Check the execution of the calibration scenario with a meshed output."""
    outputs = [
        CalibrationMetricSettings(output_name="y", metric_name="MSE"),
        CalibrationMetricSettings(output_name="z", mesh_name="mesh", metric_name="ISE"),
    ]
    calibration = CalibrationScenario(Model(), "x", outputs, calibration_space)
    calibration.execute(
        algo_name="NLOPT_COBYLA", reference_data=reference_data, max_iter=100
    )

    assert calibration.posterior_parameters["a"][0] == pytest.approx(2.0, 0.1)
    assert calibration.posterior_parameters["b"][0] == pytest.approx(3.0, 0.1)
    assert (
        calibration.formulation.optimization_problem.objective.name
        == "0.5*MSE[y]+0.5*ISE[z[mesh]]"
    )
