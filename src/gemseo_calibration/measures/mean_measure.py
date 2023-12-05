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
"""A module to compute the mean measure between the model and reference data."""

from __future__ import annotations

from numpy import nanmean

from gemseo_calibration.measure import CalibrationMeasure
from gemseo_calibration.measure import DataType


class MeanMeasure(CalibrationMeasure):
    """An abstract mean measure between the model and reference output data."""

    def __call__(self, model_dataset: DataType) -> float:  # noqa: D102
        model_data = model_dataset[self.output_name]
        self._update_bounds(model_data)
        return nanmean(self._compare_data(self._reference_data, model_data))
