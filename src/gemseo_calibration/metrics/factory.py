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
"""A factory of calibration metrics."""

from __future__ import annotations

from gemseo.core.base_factory import BaseFactory

from gemseo_calibration.metrics.base_calibration_metric import BaseCalibrationMetric
from gemseo_calibration.metrics.base_integrated_metric import BaseIntegratedMetric


class CalibrationMetricFactory(BaseFactory):
    """A factory of calibration metrics."""

    _CLASS = BaseCalibrationMetric
    _PACKAGE_NAMES = ("gemseo_calibration.metrics",)

    def is_integrated_metric(self, name: str) -> bool:
        """Return whether a calibration metric is an integrated metric.

        See
        [BaseIntegratedMetric][gemseo_calibration.metrics.base_integrated_metric.BaseIntegratedMetric].

        Args:
            name: The name of the calibration metric.

        Returns:
            Whether the calibration metric is an integrated metric.
        """
        return issubclass(self.get_class(name), BaseIntegratedMetric)
