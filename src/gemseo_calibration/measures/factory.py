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
"""A factory of calibration measures."""

from __future__ import annotations

from gemseo.core.base_factory import BaseFactory

from gemseo_calibration.measure import CalibrationMeasure
from gemseo_calibration.measures.integrated_measure import IntegratedMeasure


class CalibrationMeasureFactory(BaseFactory):
    """A factory of calibration measures."""

    _CLASS = CalibrationMeasure
    _PACKAGE_NAMES = ("gemseo_calibration.measures",)

    def is_integrated_measure(self, name: str) -> bool:
        """Return whether a calibration measure is an integrated measure.

        See
        [IntegratedMeasure][gemseo_calibration.measures.integrated_measure.IntegratedMeasure].

        Args:
            name: The name of the class of the calibration measure.

        Returns:
            Whether the calibration measure is an integrated measure.
        """
        return issubclass(self.get_class(name), IntegratedMeasure)

    @property
    def measures(self) -> list[str]:
        """The names of the available calibration measures."""
        return self.class_names
