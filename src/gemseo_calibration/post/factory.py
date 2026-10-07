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
"""A factory to post-process a `CalibrationScenario`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.optimization import OptimizationProblem
from gemseo.post.factory import PostFactory

from gemseo_calibration.post_processor import CalibrationPostProcessor

if TYPE_CHECKING:
    from gemseo.dataset import Dataset
    from gemseo.post.core.base_post_settings import BasePostSettings


class CalibrationPostFactory(PostFactory):
    """A factory for calibration post-processing."""

    _class = CalibrationPostProcessor
    _MODULE_NAMES = ("gemseo_calibration.post",)

    def execute(
        self,
        opt_problem: str | OptimizationProblem,
        reference_data: Dataset,
        prior_model_data: Dataset,
        posterior_model_data: Dataset,
        settings: BasePostSettings,
    ) -> CalibrationPostProcessor:
        """Compute the post-processing.

        Args:
            opt_problem: The optimization problem containing the data to post-process.
            reference_data: The reference data used during the calibration stage.
            prior_model_data: The model data before the calibration stage.
            posterior_model_data: The model data after the calibration stage.
            settings: The post-processor settings.

        Returns:
            The executed post-processing of the optimization problem.
        """
        if isinstance(opt_problem, str):
            opt_problem = OptimizationProblem.from_hdf(opt_problem)
        post = self.create(
            settings.target_class_name,
            opt_problem,
            reference_data,
            prior_model_data,
            posterior_model_data,
        )
        post.execute(settings)
        return post

    def create(
        self,
        post_name: str,
        opt_problem: OptimizationProblem,
        reference_data: Dataset,
        prior_model_data: Dataset,
        posterior_model_data: Dataset,
    ) -> CalibrationPostProcessor:
        """Create the post-processing.

        Args:
            opt_problem: The optimization problem containing the data to post-process.
            reference_data: The reference data used during the calibration stage.
            prior_model_data: The model data before the calibration stage.
            posterior_model_data: The model data after the calibration stage.
            post_name: The name of the post-processing method.

        Returns:
            The post-processing of the optimization problem.
        """
        return super().create(
            post_name,
            reference_data=reference_data,
            prior_model_data=prior_model_data,
            posterior_model_data=posterior_model_data,
            opt_problem=opt_problem,
        )
