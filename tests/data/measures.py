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
"""Dummy calibration metrics used for tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo_calibration.metrics.base_calibration_metric import BaseCalibrationMetric

if TYPE_CHECKING:
    from numpy import ndarray


class MetricCstr(BaseCalibrationMetric):
    """The calibration metric to be used as a constraint."""

    def _evaluate_metric(  # noqa: D102
        self,
        model_dataset: dict[str, ndarray],
    ) -> float:
        return model_dataset["y"][1, 0]


class MetricObj(BaseCalibrationMetric):
    """The calibration metric to be used as an objective."""

    maximize = True

    def _evaluate_metric(  # noqa: D102
        self,
        model_dataset: dict[str, ndarray],
    ) -> float:
        return model_dataset["y"][0, 0]


class NewCalibrationMetric(BaseCalibrationMetric):
    """The calibration metric returning zero."""

    def _evaluate_metric(  # noqa: D102
        self,
        model_dataset: dict[str, ndarray],
    ) -> float:
        return 0.0
