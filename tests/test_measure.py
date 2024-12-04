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
"""Test the CalibrationMetric and the CalibrationMetricFactory."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from numpy import array
from numpy.testing import assert_equal

if TYPE_CHECKING:
    from gemseo_calibration.metrics.base_calibration_metric import BaseCalibrationMetric


@pytest.fixture
def metric(metric_factory) -> BaseCalibrationMetric:
    """A calibration metric related to y and returning zero."""
    return metric_factory.create("NewCalibrationMetric", output_name="y")


def test_metric_init(metric):
    """Test the initialization of a CalibrationMetric."""
    assert metric.output_name == "y"
    assert metric._reference_data == []


def test_metric_set_reference_data(metric):
    """Test the method set_reference_data of CalibrationMetric."""
    dataset = {"y": array([[2.0], [4.0]])}
    metric.set_reference_data(dataset)
    assert_equal(metric._reference_data, dataset["y"])


def test_call(metric):
    """Test the method __call__ of CalibrationMetric."""
    assert metric.func("mock") == 0.0


def test_factory_create(metric_factory):
    """Test the method create() of the CalibrationMetricFactory."""
    assert (
        metric_factory.create(
            "NewCalibrationMetric", output_name="y"
        ).__class__.__name__
        == "NewCalibrationMetric"
    )

    with pytest.raises(ImportError):
        metric_factory.create("foo")


def test_factory_is_available(metric_factory):
    """Test the method is_available() of the CalibrationMetricFactory."""
    assert metric_factory.is_available("NewCalibrationMetric")
    assert not metric_factory.is_available("foo")


def test_factory_is_integrated_metric(metric_factory):
    """Test the method is_integrated_metric()."""
    assert metric_factory.is_integrated_metric("ISE")
    assert not metric_factory.is_integrated_metric("MSE")
