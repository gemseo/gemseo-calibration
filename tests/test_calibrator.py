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
"""Test the class Calibrator."""

from __future__ import annotations

import pytest
from numpy import array
from numpy.testing import assert_equal

from gemseo_calibration.calibrator import Calibrator
from gemseo_calibration.metrics.settings import CalibrationMetricSettings

CSTR_NAME = "0.5*MetricCstr[y]+0.5*MetricCstr[z]"


@pytest.fixture
def adapter(metric_factory, discipline) -> Calibrator:
    """The calibration adapter to compute calibration metrics from reference data."""
    discipline = Calibrator(
        discipline,
        ["x"],
        CalibrationMetricSettings(output_name="y", metric_name="MetricObj"),
        ["a", "b"],
    )
    discipline.default_input_data = {"a": array([0.5]), "b": array([0.5])}
    discipline.add_metric([
        CalibrationMetricSettings(output_name="y", metric_name="MetricCstr"),
        CalibrationMetricSettings(output_name="z", metric_name="MetricCstr"),
    ])
    return discipline


def test_init_objective(adapter):
    """Check the value of the objective name after initialization."""
    assert adapter.name == "Calibrator"
    assert adapter.objective_name == "MetricObj[y]"


def test_init_metrics(adapter):
    """Check the value of the metrics after initialization."""
    names_to_metrics = adapter._Calibrator__names_to_metrics
    assert str(names_to_metrics["MetricObj[y]"]) == "MetricObj[y]"
    assert str(names_to_metrics[CSTR_NAME]) == CSTR_NAME


def test_init_data(adapter):
    """Check that there is no reference data after initialization."""
    assert adapter.reference_data == {}


def test_init_grammars(adapter):
    """Check the value of the input and output grammars after initialization."""
    assert sorted(adapter.io.input_grammar.names) == ["a", "b"]
    assert sorted(adapter.io.output_grammar.names) == [
        "0.5*MetricCstr[y]+0.5*MetricCstr[z]",
        "MetricObj[y]",
    ]


def test_maximize(adapter):
    """Check the property 'maximize_objective_metric' after initialization."""
    assert adapter.maximize_objective_metric is True


def test_set_reference_data(adapter, reference_data):
    """Check that the reference data are correctly passed."""
    adapter.set_reference_data(reference_data)
    assert adapter.scenario._settings.algo_name == "CustomDOE"
    assert adapter.scenario._settings.algo_settings["samples"].shape == (
        2,
        1,
    )
    for metric in adapter._Calibrator__metrics:
        assert len(metric._reference_data) == 2

    assert_equal(adapter.reference_data, reference_data)


def test_execute_default(adapter, reference_data):
    """Check the execution of the Calibrator with default input data."""
    adapter.set_reference_data(reference_data)
    adapter.execute()
    assert adapter.io.data["MetricObj[y]"][0] == 0.25
    assert adapter.io.data[CSTR_NAME][0] == 0.5


def test_execute(adapter, reference_data):
    """Check the execution of the Calibrator with custom input data."""
    adapter.set_reference_data(reference_data)
    adapter.execute({"a": array([0.25])})
    assert adapter.io.data["MetricObj[y]"][0] == 0.125
    assert adapter.io.data[CSTR_NAME][0] == 0.25
    adapter.execute({"a": array([0.75])})
    assert adapter.io.data["MetricObj[y]"][0] == 0.375
    assert adapter.io.data[CSTR_NAME][0] == 0.75
