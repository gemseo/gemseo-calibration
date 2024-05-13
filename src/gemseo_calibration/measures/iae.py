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
"""Compute the integrated absolute error between the model and reference output data."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo_calibration.measures.integrated_measure import IntegratedMeasure

if TYPE_CHECKING:
    from gemseo.typing import RealArray


class IAE(IntegratedMeasure):
    """The integrated absolute error between the model and reference output data."""

    @staticmethod
    def _compare_data(data: RealArray, other_data: RealArray) -> RealArray:
        return abs(data - other_data)
