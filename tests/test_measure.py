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
"""Test the CalibrationMeasure and the CalibrationMeasureFactory."""
from __future__ import annotations

from pathlib import Path

import pytest
from gemseo.core.dataset import Dataset
from gemseo_calibration.measure import CalibrationMeasure
from gemseo_calibration.measures.factory import CalibrationMeasureFactory
from numpy import array
from numpy.testing import assert_equal

DATA = Path(__file__).parent / "data"


@pytest.fixture
def factory(monkeypatch) -> CalibrationMeasureFactory:
    """The factory to create a CalibrationMeasure."""
    monkeypatch.setenv("GEMSEO_PATH", DATA)
    return CalibrationMeasureFactory()


@pytest.fixture
def measure(factory) -> CalibrationMeasure:
    """A calibration measure related to y and returning zero."""
    return factory.create("NewCalibrationMeasure", output_name="y")


def test_measure_init(measure):
    """Test the initialization of a CalibrationMeasure."""
    assert measure.output_name == "y"
    assert measure._reference_data == []


def test_measure_set_reference_data(measure):
    """Test the method set_reference_data of CalibrationMeasure."""
    dataset = Dataset(by_group=False)
    dataset.add_variable("y", array([[2.0], [4.0]]))
    measure.set_reference_data(dataset)
    assert_equal(measure._reference_data, dataset.data["y"])


def test_call(measure):
    """Test the method __call__ of CalibrationMeasure."""
    assert measure("mock") == 0.0


def test_factory_create(factory, monkeypatch):
    """Test the method create() of the CalibrationMeasureFactory."""
    monkeypatch.setenv("GEMSEO_PATH", DATA)
    assert (
        factory.create("NewCalibrationMeasure", output_name="y").__class__.__name__
        == "NewCalibrationMeasure"
    )

    with pytest.raises(ImportError):
        factory.create("foo")


def test_factory_is_available(factory):
    """Test the method is_available() of the CalibrationMeasureFactory."""
    assert factory.is_available("NewCalibrationMeasure")
    assert not factory.is_available("foo")


def test_factory_measures(factory):
    """Test the property measures of the CalibrationMeasureFactory."""
    assert "NewCalibrationMeasure" in factory.measures
