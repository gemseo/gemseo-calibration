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
from gemseo.space import RandomSpace
from gemseo.discipline import Discipline
from numpy import array
from numpy import linspace

from gemseo_calibration.metrics.settings import CalibrationMetricSettings
from gemseo_calibration.scenario import CalibrationScenario
from gemseo.optimization import NLOPT_COBYLA_Settings

if TYPE_CHECKING:
    from gemseo.util.typing import StrKeyMapping
    from numpy import ndarray


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
        # TODO(bump-gemseo): IO.data is deprecated and returns a copy of the input and output data, so setting, updating or removing an item through it has no effect, and an output that _run produces through it is missing, even when produced by changing an input in place; return the outputs from _run or write them to output_data (or update_output_data(data)), write the inputs to input_data, and read input_data, output_data or get_merged_data()  # noqa: E501
        x_input = self.io.data["x"]
        # TODO(bump-gemseo): IO.data is deprecated and returns a copy of the input and output data, so setting, updating or removing an item through it has no effect, and an output that _run produces through it is missing, even when produced by changing an input in place; return the outputs from _run or write them to output_data (or update_output_data(data)), write the inputs to input_data, and read input_data, output_data or get_merged_data()  # noqa: E501
        a_parameter = self.io.data["a"]
        # TODO(bump-gemseo): IO.data is deprecated and returns a copy of the input and output data, so setting, updating or removing an item through it has no effect, and an output that _run produces through it is missing, even when produced by changing an input in place; return the outputs from _run or write them to output_data (or update_output_data(data)), write the inputs to input_data, and read input_data, output_data or get_merged_data()  # noqa: E501
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
        # TODO(bump-gemseo): IO.data is deprecated and returns a copy of the input and output data, so setting, updating or removing an item through it has no effect, and an output that _run produces through it is missing, even when produced by changing an input in place; return the outputs from _run or write them to output_data (or update_output_data(data)), write the inputs to input_data, and read input_data, output_data or get_merged_data()  # noqa: E501
        x_input = self.io.data["x"]
        z_mesh = linspace(0, 1, 5)
        y_output = 2 * x_input * z_mesh
        z_output = 3 * x_input[0] * z_mesh
        self.io.update_output_data({"y": y_output, "z": z_output, "mesh": z_mesh})


@pytest.fixture(scope="module")
def calibration_space() -> RandomSpace:
    """The calibration space."""
    space = RandomSpace()
    # TODO(bump-gemseo): a RandomSpace holds only random variables; add a deterministic variable to a DesignSpace instead  # noqa: E501
    space.add_variable("a", lower_bound=0.0, upper_bound=10.0, value=0.0)
    # TODO(bump-gemseo): a RandomSpace holds only random variables; add a deterministic variable to a DesignSpace instead  # noqa: E501
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
        algorithm_settings=NLOPT_COBYLA_Settings(reference_data=reference_data, max_iter=100
    ))

    assert calibration.posterior_parameters["a"][0] == pytest.approx(2.0, 0.1)
    assert calibration.posterior_parameters["b"][0] == pytest.approx(3.0, 0.1)
    assert (
        calibration.formulation.problem.objective.name
        == "0.5*MSE[y]+0.5*ISE[z[mesh]]"
    )
