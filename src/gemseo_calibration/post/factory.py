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
"""A factory to post-process  a :class:`.CalibrationScenario`."""
from __future__ import annotations

import logging

from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.core.dataset import Dataset
from gemseo.core.factory import Factory
from gemseo.post.post_factory import PostFactory

from gemseo_calibration.post_processor import CalibrationPostProcessor

LOGGER = logging.getLogger(__name__)


class CalibrationPostFactory(PostFactory):
    """A factory for calibration post-processing."""

    def __init__(self) -> None:  # noqa: D107
        self.factory = Factory(CalibrationPostProcessor, ("gemseo_calibration.post",))
        self.executed_post = []

    def execute(
        self,
        opt_problem: str | OptimizationProblem,
        reference_data: Dataset,
        prior_model_data: Dataset,
        posterior_model_data: Dataset,
        post_name: str,
        **options,
    ) -> CalibrationPostProcessor:
        """Compute the post-processing.

        Args:
            opt_problem: The optimization problem containing the data to post-process.
            reference_data: The reference data used during the calibration stage.
            prior_model_data: The model data before the calibration stage.
            posterior_model_data: The model data after the calibration stage.
            post_name: The name of the post-processing method.
            **options: The options of the post-processing method.

        Returns:
            The executed post-processing of the optimization problem.
        """
        if isinstance(opt_problem, str):
            opt_problem = OptimizationProblem.import_hdf(opt_problem)
        post = self.create(
            opt_problem,
            reference_data,
            prior_model_data,
            posterior_model_data,
            post_name,
        )
        post.execute(**options)
        self.executed_post.append(post)
        return post

    def create(
        self,
        opt_problem: OptimizationProblem,
        reference_data: Dataset,
        prior_model_data: Dataset,
        posterior_model_data: Dataset,
        post_name: str,
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
        return self.factory.create(
            post_name,
            reference_data=reference_data,
            prior_model_data=prior_model_data,
            posterior_model_data=posterior_model_data,
            opt_problem=opt_problem,
        )
