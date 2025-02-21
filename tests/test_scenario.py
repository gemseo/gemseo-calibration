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
"""Test the class CalibrationScenario."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.disciplines.auto_py import AutoPyDiscipline
from gemseo.post.opt_history_view import OptHistoryView
from numpy import array

from gemseo_calibration.calibrator import Calibrator
from gemseo_calibration.metrics.settings import CalibrationMetricSettings
from gemseo_calibration.post.data_versus_model.post import DataVersusModel
from gemseo_calibration.scenario import CalibrationScenario

if TYPE_CHECKING:
    from gemseo.core.discipline.discipline import Discipline


@pytest.fixture(scope="module")
def calibration_space() -> DesignSpace:
    """The space of the parameters to calibrate."""
    space = DesignSpace()
    space.add_variable("a", lower_bound=0.0, upper_bound=1.0, value=0.5)
    space.add_variable("b", lower_bound=0.0, upper_bound=1.0, value=0.5)
    return space


@pytest.fixture
def calibration_scenario(
    metric_factory,
    discipline: Discipline,
    calibration_space: DesignSpace,
) -> CalibrationScenario:
    """The scenario to calibrate the discipline with the reference input data."""
    scenario = CalibrationScenario(
        discipline,
        "x",
        CalibrationMetricSettings(output_name="y", metric_name="MetricObj"),
        calibration_space,
        name="calib",
    )
    scenario.add_constraint(
        [
            CalibrationMetricSettings(output_name="y", metric_name="MetricCstr"),
            CalibrationMetricSettings(output_name="z", metric_name="MetricCstr"),
        ],
        "ineq",
        value=0.05,
    )
    return scenario


def test_init(calibration_scenario):
    """Check the initialization of the CalibrationScenario + add of the constraint."""
    assert calibration_scenario.formulation_name == "DisciplinaryOpt"
    assert len(calibration_scenario.disciplines) == 1
    assert isinstance(calibration_scenario.disciplines[0], Calibrator)
    assert calibration_scenario.design_space.variable_names == ["a", "b"]
    assert calibration_scenario.name == "calib"
    assert isinstance(calibration_scenario.calibrator, Calibrator)
    assert (
        calibration_scenario.formulation.optimization_problem.minimize_objective
        is False
    )


@pytest.mark.parametrize("list_of_disciplines", [False, True])
@pytest.mark.parametrize("list_of_inputs", [False, True])
@pytest.mark.parametrize("list_of_outputs", [False, True])
@pytest.mark.parametrize("list_of_constraints", [False, True])
def test_init_list(
    metric_factory,
    discipline,
    calibration_space,
    list_of_disciplines,
    list_of_inputs,
    list_of_outputs,
    list_of_constraints,
):
    """Check the instantiation of a scenario with or without lists of variables."""
    disciplines = [discipline] if list_of_disciplines else discipline
    inputs = ["x"] if list_of_inputs else "x"
    output = CalibrationMetricSettings(output_name="y", metric_name="MetricObj")
    outputs = [output] if list_of_outputs else output
    constraint = CalibrationMetricSettings(output_name="z", metric_name="MetricCstr")
    constraints = [constraint] if list_of_constraints else constraint
    scenario = CalibrationScenario(disciplines, inputs, outputs, calibration_space)
    scenario.add_constraint(constraints)
    assert scenario.calibrator.scenario.design_space.variable_names == ["x"]
    assert scenario.calibrator.scenario.formulation._objective_name == "y"


def test_calibration_adapter(calibration_scenario):
    """Check the calibrator after initialization + add of the constraint."""
    names_to_metrics = calibration_scenario.calibrator._Calibrator__names_to_metrics
    assert sorted(names_to_metrics.keys()) == [
        "0.5*MetricCstr[y]+0.5*MetricCstr[z]",
        "MetricObj[y]",
    ]


def test_constraint(calibration_scenario):
    """Test the constraint added to the inner optimization problem.

    with the reference input data.
    """
    constraints = calibration_scenario.formulation.optimization_problem.constraints
    assert len(constraints) == 1
    assert str(constraints[0]) == "0.5*MetricCstr[y]+0.5*MetricCstr[z](a, b) <= 0.05"


def test_execute(calibration_scenario, reference_data):
    """Test that the reference data are correctly passed during the execution."""
    calibration_scenario.execute(
        algo_name="NLOPT_COBYLA", reference_data=reference_data, max_iter=10
    )
    assert calibration_scenario.prior_parameters == {
        "a": array([0.5]),
        "b": array([0.5]),
    }
    assert (
        calibration_scenario.posterior_parameters
        != calibration_scenario.prior_parameters
    )
    assert set(calibration_scenario.posterior_parameters.keys()) == {"a", "b"}


def test_posts(calibration_scenario):
    """Check the list of available post-processings."""
    posts = calibration_scenario.posts
    assert "OptHistoryView" in posts
    assert "DataVersusModel" in posts


def test_post_process(calibration_scenario, reference_data):
    """Check the post-processing of a calibration scenario."""
    calibration_scenario.execute(
        algo_name="NLOPT_COBYLA", reference_data=reference_data, max_iter=2
    )
    post = calibration_scenario.post_process(
        post_name="OptHistoryView", save=False, show=False
    )
    assert isinstance(post, OptHistoryView)
    post = calibration_scenario.post_process(
        post_name="DataVersusModel", output="y", save=False, show=False
    )
    assert isinstance(post, DataVersusModel)


def f(x: float, p: float) -> float:
    y = x + p
    return y  # noqa: RET504


def test_float_calibration_parameter():
    """Check that CalibrationScenario supports disciplines with float arguments."""
    discipline = AutoPyDiscipline(f)

    calibration_space = DesignSpace()
    calibration_space.add_variable("p", lower_bound=-1.0, upper_bound=1.0, value=-1.0)

    reference_data = {
        "x": array([[0.5], [1.0]]),
        "y": array([[1.0], [1.5]]),
    }

    scenario = CalibrationScenario(
        discipline,
        "x",
        CalibrationMetricSettings(output_name="y", metric_name="MSE"),
        calibration_space,
    )
    scenario.execute(
        algo_name="NLOPT_COBYLA", reference_data=reference_data, max_iter=10
    )
    assert scenario.optimization_result.x_opt == 0.5
