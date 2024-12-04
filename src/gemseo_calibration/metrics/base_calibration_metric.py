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
"""Base class for metrics to compare data sets."""

from __future__ import annotations

from typing import ClassVar

from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.typing import RealArray

DataType = dict[str, RealArray]
"""The type of data.

The data are set as `{variable_name: variable_values}`
where `variable_values` is a 2D NumPy array
whose rows are the samples and columns are the components of the variable.
"""


class BaseCalibrationMetric(MDOFunction):
    """The base class for metrics to compare data sets."""

    output_name: str
    """The name of the output used by the metric for calibration."""

    maximize: ClassVar[bool] = False
    """Whether to maximize the calibration metric."""

    def __init__(
        self,
        output_name: str,
        name: str = "",
        f_type: MDOFunction.FunctionType = MDOFunction.FunctionType.NONE,
    ) -> None:
        """
        Args:
            output_name: The name of the output to be taken into account by the metric.
        """  # noqa: D205,D212,D415
        self.output_name = output_name
        super().__init__(
            self._evaluate_metric, name or self._compute_name(), f_type=f_type
        )
        self._reference_data = []

    @property
    def full_output_name(self) -> str:
        """The full name of the output."""
        return self.output_name

    def _compute_name(self) -> str:
        """Return the name of the metric."""
        return f"{self.__class__.__name__}({self.output_name})"

    def set_reference_data(self, reference_dataset: DataType) -> None:
        """Define the reference input-output data set.

        Args:
            reference_dataset: The reference input-output data set.
        """
        self._reference_data = reference_dataset[self.output_name]

    def _evaluate_metric(self, model_dataset: DataType) -> float:
        """Evaluate the metric given a model dataset.

        Args:
            model_dataset: The model dataset.

        Returns:
            The value of the metric.
        """
        raise NotImplementedError

    @staticmethod
    def _compare_data(data: RealArray, other_data: RealArray) -> RealArray:
        """Compare two data arrays.

        Args:
            data: The first data array.
            other_data: The second data array.

        Returns:
            The comparison between the two data arrays.
        """
        raise NotImplementedError
