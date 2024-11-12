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
"""Plot the model data versus the reference data."""

from __future__ import annotations

from typing import ClassVar

from gemseo.datasets.dataset import Dataset
from numpy import newaxis

from gemseo_calibration.post.data_versus_model.settings import DataVersusModelSettings
from gemseo_calibration.post.multiple_scatter import MultipleScatter
from gemseo_calibration.post_processor import CalibrationPostProcessor


class DataVersusModel(CalibrationPostProcessor[DataVersusModelSettings]):
    """Scatter plot of the model data versus the reference ones."""

    Settings: ClassVar[type[DataVersusModelSettings]] = DataVersusModelSettings

    def _plot(self, settings: DataVersusModelSettings) -> None:
        output = settings.output
        opt_name = f"Opt[{output}]"
        init_name = f"Init[{output}]"
        ref_name = f"Ref[{output}]"

        dataset = Dataset()
        dataset.add_variable(
            opt_name, self._posterior_model_data[output].mean(1)[:, newaxis]
        )
        dataset.add_variable(
            init_name, self._prior_model_data[output].mean(1)[:, newaxis]
        )
        dataset.add_variable(ref_name, self._reference_data[output].mean(1)[:, newaxis])
        plot = MultipleScatter(dataset, x=ref_name, y=[init_name, opt_name])
        plot.color = ["blue", "red"]
        plot.xlabel = "Reference"
        plot.ylabel = "Model"
        plot.labels = {opt_name: "After calibration", init_name: "Before calibration"}
        figures = plot.execute(save=False, show=False)
        for figure in figures:
            self._add_figure(figure)
