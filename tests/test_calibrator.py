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

from pathlib import Path

import pytest
from gemseo_calibration.calibrator import CalibrationMeasure
from gemseo_calibration.calibrator import Calibrator
from numpy import array

DATA = Path(__file__).parent / "data"

CSTR_NAME = "0.5*MeasureCstr[y]+0.5*MeasureCstr[z]"


@pytest.fixture
def adapter(monkeypatch, discipline) -> Calibrator:
    """The calibration adapter to compute calibration measures from reference data."""
    monkeypatch.setenv("GEMSEO_PATH", DATA)
    discipline = Calibrator(
        discipline, ["x"], CalibrationMeasure("y", "MeasureObj"), ["a", "b"]
    )
    discipline.default_inputs = {"a": array([0.5]), "b": array([0.5])}
    discipline.add_measure(
        [CalibrationMeasure("y", "MeasureCstr"), CalibrationMeasure("z", "MeasureCstr")]
    )
    return discipline


def test_init_objective(adapter):
    """Check the value of the objective name after initialization."""
    assert adapter.name == "Calibrator"
    assert adapter.objective_name == "MeasureObj[y]"


def test_init_measures(adapter):
    """Check the value of the measures after initialization."""
    names_to_measures = adapter._Calibrator__names_to_measures
    assert str(names_to_measures["MeasureObj[y]"]) == "MeasureObj[y]"
    assert str(names_to_measures[CSTR_NAME]) == CSTR_NAME


def test_init_data(adapter):
    """Check that there is no reference data after initialization."""
    assert adapter.reference_data is None


def test_init_grammars(adapter):
    """Check the value of the input and output grammars after initialization."""
    assert sorted(list(adapter.get_input_data_names())) == ["a", "b"]
    assert sorted(list(adapter.get_output_data_names())) == [
        "0.5*MeasureCstr[y]+0.5*MeasureCstr[z]",
        "MeasureObj[y]",
    ]


def test_maximize(adapter):
    """Check the property 'maximize_objective_measure' after initialization."""
    assert adapter.maximize_objective_measure is True


def test_set_reference_data(adapter, reference_data):
    """Check that the reference data are correctly passed."""
    adapter.set_reference_data(reference_data)
    assert adapter.reference_data.name == reference_data.name
    assert adapter.scenario.default_inputs["algo"] == "CustomDOE"
    assert adapter.scenario.default_inputs["algo_options"]["samples"].shape == (2, 1)
    for measure in adapter._Calibrator__measures:
        assert len(measure._reference_data) == 2

    assert adapter.reference_data == reference_data


def test_execute_default(adapter, reference_data):
    """Check the execution of the Calibrator with default input data."""
    adapter.set_reference_data(reference_data)
    adapter.execute()
    assert adapter.local_data["MeasureObj[y]"][0] == 0.25
    assert adapter.local_data[CSTR_NAME][0] == 0.5


def test_execute(adapter, reference_data):
    """Check the execution of the Calibrator with custom input data."""
    adapter.set_reference_data(reference_data)
    adapter.execute({"a": array([0.25])})
    assert adapter.local_data["MeasureObj[y]"][0] == 0.125
    assert adapter.local_data[CSTR_NAME][0] == 0.25
    adapter.execute({"a": array([0.75])})
    assert adapter.local_data["MeasureObj[y]"][0] == 0.375
    assert adapter.local_data[CSTR_NAME][0] == 0.75
