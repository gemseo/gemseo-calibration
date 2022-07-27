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
from __future__ import annotations

from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.core.dataset import Dataset
from gemseo.post.opt_post_processor import OptPostProcessor
from gemseo_calibration.post_processor import CalibrationPostProcessor


class NewOptPostProcessor(OptPostProcessor):
    """A new optimization post-processor."""

    pass


class NewCalibrationPostProcessor(CalibrationPostProcessor):
    """A new calibration post-processor."""

    def __init__(
        self,
        opt_problem: OptimizationProblem,
        reference_data: Dataset,
        prior_model_data: Dataset,
        posterior_model_data: Dataset,
    ):  # type: (...) -> None
        # noqa: D205 D212 D415
        """
        Args:
            opt_problem: The optimization problem to run.
            reference_data: The reference data.
            prior_model_data: The model data before the calibration.
            posterior_model_data: The model data after the calibration.
        """
        super().__init__(
            opt_problem, reference_data, prior_model_data, posterior_model_data
        )
        self.executed = False

    def execute(self, **options) -> None:  # noqa: D102
        self.executed = True
