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
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.dataset import Dataset
from gemseo.core.discipline import MDODiscipline
from gemseo_calibration.calibrator import CalibrationMeasure
from gemseo_calibration.scenario import CalibrationScenario
from numpy import array
from numpy import linspace


class Model(MDODiscipline):
    """The model to be calibrated."""

    def __init__(self):  # noqa: D107
        super().__init__()
        self.input_grammar.update(["x", "a", "b"])
        self.output_grammar.update(["y", "z", "mesh"])
        self.default_inputs = {"x": array([0.0]), "a": array([0.0]), "b": array([0.0])}

    def _run(self):  # noqa: D107
        x_input = self.local_data["x"]
        a_parameter = self.local_data["a"]
        b_parameter = self.local_data["b"]
        z_mesh = linspace(0, 1, 5)
        y_output = a_parameter * x_input * z_mesh
        z_output = b_parameter * x_input[0] * z_mesh
        self.store_local_data(y=y_output, z=z_output, mesh=z_mesh)


class ReferenceModel(MDODiscipline):
    """The model to be approximated."""

    def __init__(self):  # noqa: D107
        super().__init__()
        self.input_grammar.update(["x"])
        self.output_grammar.update(["y", "z", "mesh"])
        self.default_inputs = {"x": array([0.0])}

    def _run(self):  # noqa: D107
        x_input = self.local_data["x"]
        z_mesh = linspace(0, 1, 5)
        y_output = 2 * x_input * z_mesh
        z_output = 3 * x_input[0] * z_mesh
        self.store_local_data(y=y_output, z=z_output, mesh=z_mesh)


@pytest.fixture(scope="module")
def calibration_space() -> ParameterSpace:
    """The calibration space."""
    space = ParameterSpace()
    space.add_variable("a", l_b=0.0, u_b=10.0, value=0.0)
    space.add_variable("b", l_b=0.0, u_b=10.0, value=0.0)
    return space


@pytest.fixture(scope="module")
def reference_data() -> Dataset:
    """The reference dataset."""
    reference = ReferenceModel()
    reference.set_cache_policy("MemoryFullCache")
    reference.execute({"x": array([1.0])})
    reference.execute({"x": array([2.0])})
    return reference.cache.export_to_dataset(by_group=True)


def test_execute(reference_data, calibration_space):
    """Check the execution of the calibration scenario with a meshed output."""
    outputs = [
        CalibrationMeasure(output="y", measure="MSE"),
        CalibrationMeasure(output="z", mesh="mesh", measure="ISE"),
    ]
    calibration = CalibrationScenario(Model(), "x", outputs, calibration_space)
    calibration.execute(
        {"algo": "NLOPT_COBYLA", "reference_data": reference_data, "max_iter": 100}
    )

    assert calibration.posterior_parameters["a"][0] == pytest.approx(2.0, 0.1)
    assert calibration.posterior_parameters["b"][0] == pytest.approx(3.0, 0.1)
    assert (
        calibration.formulation.opt_problem.objective.name
        == "0.5*MSE[y]+0.5*ISE[z[mesh]]"
    )
