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

from typing import TYPE_CHECKING

from gemseo.post.base_post import BasePost

from gemseo_calibration.post_processor import CalibrationPostProcessor

if TYPE_CHECKING:
    from gemseo.algos.optimization_problem import OptimizationProblem
    from gemseo.datasets.optimization_dataset import OptimizationDataset
    from numpy import ndarray


class NewOptPostProcessor(BasePost):
    """A new optimization post-processor."""


class NewCalibrationPostProcessor(CalibrationPostProcessor):
    """A new calibration post-processor."""

    def __init__(
        self,
        opt_problem: OptimizationProblem | OptimizationDataset,
        reference_data: dict[str, ndarray],
        prior_model_data: dict[str, ndarray],
        posterior_model_data: dict[str, ndarray],
    ) -> None:
        super().__init__(
            opt_problem, reference_data, prior_model_data, posterior_model_data
        )
        self.executed = False

    def execute(self, **options) -> None:  # noqa: D102
        self.executed = True
