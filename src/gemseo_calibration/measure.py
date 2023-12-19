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
"""A module to measure the consistency or the inconsistency between two data sets."""

from __future__ import annotations

from typing import ClassVar

from gemseo.core.mdofunctions.mdo_function import MDOFunction
from numpy import inf
from numpy import nanmax
from numpy import nanmin
from numpy import ndarray

DataType = dict[str, ndarray]
"""The type of data.

The data are set as ``{variable_name: variable_values}``
where ``variable_values`` is a 2D NumPy array
whose rows are the samples and columns are the components of the variable.
"""


class CalibrationMeasure(MDOFunction):
    """A measure of the consistency (or inconsistency) between two data sets."""

    output_name: str
    """The name of the output used by the measure for calibration."""

    maximize: ClassVar[bool] = False
    """Whether to maximize the calibration measure."""

    def __init__(
        self,
        output_name: str,
        name: str = "",
        f_type: MDOFunction.FunctionType = MDOFunction.FunctionType.NONE,
    ) -> None:
        """
        Args:
            output_name: The name of the output to be taken into account by the measure.
        """  # noqa: D205,D212,D415
        self.output_name = output_name
        super().__init__(None, name or self._compute_name(), f_type=f_type)
        self._lower_bound = -inf
        self._upper_bound = inf
        self._reference_data = []

    @property
    def full_output_name(self) -> str:
        """The full name of the output."""
        return self.output_name

    def _compute_name(self) -> str:
        """Return the name of the measure."""
        return f"{self.__class__.__name__}({self.output_name})"

    def set_reference_data(self, reference_dataset: DataType) -> None:
        """Define the reference input-output data set.

        Args:
            reference_dataset: The reference input-output data set.
        """
        self._reference_data = reference_dataset[self.output_name]
        self._lower_bound = nanmin(self._reference_data)
        self._upper_bound = nanmax(self._reference_data)

    def _update_bounds(self, data: ndarray) -> None:
        """Update the lower and upper bounds of the output.

        Args:
            data: The value of the output.
        """
        self._lower_bound = min(data.min(), self._lower_bound)
        self._upper_bound = max(data.max(), self._upper_bound)

    def __call__(self, model_dataset: DataType) -> float:
        """Measure the (in)consistency between the model dataset and the reference one.

        Args:
            model_dataset: The model dataset.

        Returns:
            The measure of the (in)consistency between the model and reference datasets.
        """
        raise NotImplementedError

    @staticmethod
    def _compare_data(data: ndarray, other_data: ndarray) -> ndarray:
        """Compare two data arrays.

        Args:
            data: The first data array.
            other_data: The second data array.

        Returns:
            The comparison between the two data arrays.
        """
        raise NotImplementedError
