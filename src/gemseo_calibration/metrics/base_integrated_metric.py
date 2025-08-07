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
"""Base class for integrated metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import all as np_all
from numpy import diff
from numpy import mean

try:
    # Numpy >= 2
    from numpy import trapezoid
except ImportError:  # pragma: no cover
    # Numpy < 2
    from numpy import trapz as trapezoid

from scipy.interpolate import interp1d

from gemseo_calibration.metrics.base_calibration_metric import BaseCalibrationMetric
from gemseo_calibration.metrics.base_calibration_metric import DataType

if TYPE_CHECKING:
    from gemseo.typing import RealArray


class BaseIntegratedMetric(BaseCalibrationMetric):
    """The base class for integrated metrics."""

    mesh_name: str
    """The name of the 1D mesh."""

    def __init__(
        self,
        output_name: str,
        mesh_name: str,
        name: str = "",
        f_type: BaseCalibrationMetric.FunctionType = BaseCalibrationMetric.FunctionType.NONE,  # noqa: E501
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

    def _evaluate_metric(self, model_dataset: DataType) -> float:  # noqa: D102
        """Evaluate the metric by comparing reference data interpolated on model data.

        The interpolation accepts reference abscissa or model abscissa having
        monotonically increasing or decreasing values. Extrapolation is forbidden.
        """
        model_data = model_dataset[self.output_name]
        model_mesh = model_dataset[self.mesh_name]
        compared_data = []
        for x_ref, y_ref, x_model, y_model in zip(
            self.__reference_mesh, self._reference_data, model_mesh, model_data
        ):
            if np_all(diff(x_ref) < 0):
                x_ref = x_ref[::-1]
                y_ref = y_ref[::-1]
            if np_all(diff(x_model) < 0):
                x_model = x_model[::-1]
                y_model = y_model[::-1]

            interpolator = interp1d(
                x_model,
                y_model,
                assume_sorted=True,
                bounds_error=True,
            )
            compared_data.append(
                self._compare_data(
                    y_ref,
                    interpolator(x_ref),
                )
            )
        return mean([
            trapezoid(
                data,
                x_ref,
            )
            for data in compared_data
        ])

    @property
    def reference_mesh(self) -> RealArray:
        """The reference mesh."""
        return self.__reference_mesh

    @property
    def full_output_name(self) -> str:  # noqa: D102
        return f"{self.output_name}[{self.mesh_name}]"

    def set_reference_data(self, reference_dataset: DataType) -> None:  # noqa: D102
        self.__reference_mesh = reference_dataset[self.mesh_name]
        super().set_reference_data(reference_dataset)
