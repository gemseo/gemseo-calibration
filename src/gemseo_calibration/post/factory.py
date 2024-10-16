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
from typing import Any

from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.post.factory import PostFactory

from gemseo_calibration.post_processor import CalibrationPostProcessor

if TYPE_CHECKING:
    from pathlib import Path

    from gemseo.datasets.dataset import Dataset


class CalibrationPostFactory(PostFactory):
    """A factory for calibration post-processing."""

    _CLASS = CalibrationPostProcessor
    _MODULE_NAMES = ("gemseo_calibration.post",)

    def execute(
        self,
        opt_problem: str | OptimizationProblem,
        reference_data: Dataset,
        prior_model_data: Dataset,
        posterior_model_data: Dataset,
        post_name: str,
        save: bool = True,
        show: bool = False,
        file_path: str | Path = "",
        directory_path: str | Path = "",
        file_name: str = "",
        file_extension: str = "",
        **options: Any,
    ) -> CalibrationPostProcessor:
        """Compute the post-processing.

        Args:
            opt_problem: The optimization problem containing the data to post-process.
            reference_data: The reference data used during the calibration stage.
            prior_model_data: The model data before the calibration stage.
            posterior_model_data: The model data after the calibration stage.
            post_name: The name of the post-processing method.
            save: Whether to save the figure.
            show: Whether to display the figure.
            file_path: The path of the file to save the figures.
                If the extension is missing, use `file_extension`.
                If empty,
                create a file path
                from `directory_path`, `file_name` and `file_extension`.
            directory_path: The path of the directory to save the figures.
                If empty, use the current working directory.
            file_name: The name of the file to save the figures.
                If empty, use a default one generated by the post-processing.
            file_extension: A file extension, e.g. 'png', 'pdf', 'svg', ...
                If empty, use a default file extension.
            **options: The options of the post-processor.

        Returns:
            The executed post-processing of the optimization problem.
        """
        if isinstance(opt_problem, str):
            opt_problem = OptimizationProblem.from_hdf(opt_problem)
        post = self.create(
            post_name,
            opt_problem,
            reference_data,
            prior_model_data,
            posterior_model_data,
        )
        post.execute(
            save=save,
            show=show,
            file_path=file_path,
            directory_path=directory_path,
            file_name=file_name,
            file_extension=file_extension,
            **options,
        )
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
