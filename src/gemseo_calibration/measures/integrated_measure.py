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
"""A module to compute the integrated measure between two data sets."""

from __future__ import annotations

from numpy import interp
from numpy import mean
from numpy import trapz as integrate

from gemseo_calibration.measure import CalibrationMeasure
from gemseo_calibration.measure import DataType


class IntegratedMeasure(CalibrationMeasure):
    """An abstract integrated measure between two output data sets."""

    mesh_name: str
    """The name of the 1D mesh."""

    def __init__(
        self,
        output_name: str,
        mesh_name: str,
        name: str = "",
        f_type: CalibrationMeasure.FunctionType = CalibrationMeasure.FunctionType.NONE,
    ) -> None:
        """
        Args:
            mesh_name: The name of the 1D mesh.
        """  # noqa: D205 D212 D415
        self.mesh_name = mesh_name
        self.__reference_mesh = None
        super().__init__(output_name, name=name, f_type=f_type)

    def _compute_name(self) -> str:
        return f"{self.__class__.__name__}({self.output_name};{self.mesh_name})"

    def __call__(self, model_dataset: DataType) -> float:  # noqa: D102
        model_data = model_dataset[self.output_name]
        model_mesh = model_dataset[self.mesh_name]
        self._update_bounds(model_data)
        return mean([
            integrate(
                self._compare_data(
                    self._reference_data[i],
                    interp(self.__reference_mesh[i], model_mesh[i], model_data[i]),
                ),
                self.__reference_mesh[i],
            )
            for i in range(len(model_data))
        ])

    @property
    def full_output_name(self) -> str:  # noqa: D102
        return f"{self.output_name}[{self.mesh_name}]"

    def set_reference_data(self, reference_dataset: DataType) -> None:  # noqa: D102
        self.__reference_mesh = reference_dataset[self.mesh_name]
        super().set_reference_data(reference_dataset)
