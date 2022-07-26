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
"""A factory to create instances of :class:`.CalibrationMeasure`."""
from __future__ import annotations

from typing import Any

from gemseo.core.factory import Factory

from gemseo_calibration.measure import CalibrationMeasure
from gemseo_calibration.measures.integrated_measure import IntegratedMeasure


class CalibrationMeasureFactory:
    """A factory to create instances of :class:`.CalibrationMeasure`."""

    is_integrated_measure: bool
    """Whether the calibration measure is an :class:`.IntegratedMeasure`."""

    def __init__(self) -> None:  # noqa: D107,D415,D102,D417
        self.__factory = Factory(CalibrationMeasure, ("gemseo_calibration.measures",))
        factory = Factory(IntegratedMeasure, ("gemseo_calibration.measures",))
        self.is_integrated_measure = factory.is_available

    def create(self, name: str, **options: Any) -> CalibrationMeasure:
        """Instantiate a :class:`.CalibrationMeasure` from its class name.

        Args:
            name: The name of a class inheriting from :class:`.CalibrationMeasure`.
            **options: The options of the calibration measure.

        Returns:
            The calibration measure.
        """
        return self.__factory.create(name, **options)

    @property
    def measures(self) -> list[str]:
        """The names of the available calibration measures."""
        return self.__factory.classes

    def is_available(
        self,
        name: str,
    ) -> bool:
        """Return whether a calibration measure is available.

        Args:
            name: The name of a calibration measure.

        Returns:
            Whether the measure is available.
        """
        return self.__factory.is_available(name)
