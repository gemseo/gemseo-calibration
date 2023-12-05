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
"""Base class for all calibration post-processing methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.post.opt_post_processor import OptPostProcessor

if TYPE_CHECKING:
    from gemseo.algos.opt_problem import OptimizationProblem

    from gemseo_calibration.measure import DataType


class CalibrationPostProcessor(OptPostProcessor):
    """Abstract class for optimization post-processing methods."""

    def __init__(
        self,
        opt_problem: OptimizationProblem,
        reference_data: DataType,
        prior_model_data: DataType,
        posterior_model_data: DataType,
    ) -> None:
        """
        Args:
            opt_problem: The optimization problem to run.
            reference_data: The reference data.
            prior_model_data: The model data before the calibration.
            posterior_model_data: The model data after the calibration.
        """  # noqa: D205, D212, D415
        super().__init__(opt_problem)
        self._reference_data = reference_data
        self._prior_model_data = prior_model_data
        self._posterior_model_data = posterior_model_data
