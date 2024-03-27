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

from typing import TYPE_CHECKING

import pytest
from numpy import array
from numpy.testing import assert_equal

if TYPE_CHECKING:
    from gemseo_calibration.measure import CalibrationMeasure


@pytest.fixture()
def measure(measure_factory) -> CalibrationMeasure:
    """A calibration measure related to y and returning zero."""
    return measure_factory.create("NewCalibrationMeasure", output_name="y")


def test_measure_init(measure):
    """Test the initialization of a CalibrationMeasure."""
    assert measure.output_name == "y"
    assert measure._reference_data == []


def test_measure_set_reference_data(measure):
    """Test the method set_reference_data of CalibrationMeasure."""
    dataset = {"y": array([[2.0], [4.0]])}
    measure.set_reference_data(dataset)
    assert_equal(measure._reference_data, dataset["y"])


def test_call(measure):
    """Test the method __call__ of CalibrationMeasure."""
    assert measure("mock") == 0.0


def test_factory_create(measure_factory):
    """Test the method create() of the CalibrationMeasureFactory."""
    assert (
        measure_factory.create(
            "NewCalibrationMeasure", output_name="y"
        ).__class__.__name__
        == "NewCalibrationMeasure"
    )

    with pytest.raises(ImportError):
        measure_factory.create("foo")


def test_factory_is_available(measure_factory):
    """Test the method is_available() of the CalibrationMeasureFactory."""
    assert measure_factory.is_available("NewCalibrationMeasure")
    assert not measure_factory.is_available("foo")


def test_factory_is_integrated_measure(measure_factory):
    """Test the method is_integrated_measure()."""
    assert measure_factory.is_integrated_measure("ISE")
    assert not measure_factory.is_integrated_measure("MSE")


def test_factory_measures(measure_factory):
    """Test the property measures of the CalibrationMeasureFactory."""
    assert "NewCalibrationMeasure" in measure_factory.measures
