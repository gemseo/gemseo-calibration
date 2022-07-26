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
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
r"""Overlay several scatter plots from a :class:`.Dataset`.

A :class:`Scatter` plot
represents a set of points :math:`\{x_i,y_i\}_{1\leq i \leq n}`
as markers on a classical plot,
while a :class:`MultipleScatter` plot
represents a set of points :math:`\{x_i,y_{i,1},\ldots,y_{i,d}\}_{1\leq i \leq n}`
as markers on a classical plot,
with one color per series :math:`\{y_i\}_{1\leq i \leq n}`.
"""
from __future__ import annotations

from typing import Iterable
from typing import Mapping

from gemseo.core.dataset import Dataset
from gemseo.post.dataset.dataset_plot import DatasetPlot
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class MultipleScatter(DatasetPlot):
    """Overlay several scatter y_i versus x."""

    def __init__(
        self,
        dataset: Dataset,
        x: str,
        y: str | Iterable[str],
        x_comp: str = 0,
        y_comp: Mapping[str, int] = None,
    ) -> None:
        # noqa: D205 D212 D415
        """
        Args:
            x: The name of the variable on the x-axis.
            y: The names of the variables on the y-axis.
            x_comp: The component of x.
            y_comp: The components of y,
                where the names are the names of the variables
                and the values are the components.
                If None or if a name is missing,
                use the first component.
        """
        super().__init__(dataset=dataset, x=x, y=y, x_comp=x_comp, y_comp=y_comp)

    def _plot(
        self,
        fig: None | Figure = None,
        axes: None | Axes = None,
    ) -> list[Figure]:
        x = self._param.x
        y = self._param.y
        if isinstance(y, str):
            y = [y]

        y_comp = self._param.y_comp or {}
        for name in y:
            y_comp[name] = y_comp.get(name, 0)

        reference = self.dataset.get_data_by_names(x, False)[:, self._param.x_comp]

        fig, axes = self._get_figure_and_axes(fig, axes)
        bounds = [min(reference), max(reference)]
        axes.plot(bounds, bounds, color="gray", linestyle="--", marker="o")
        self._set_color(len(y))
        for index, name in enumerate(y):
            axes.plot(
                reference,
                self.dataset.get_data_by_names(name, False)[:, y_comp[name]],
                color=self.color[index],
                marker="o",
                linestyle="",
                label=self.labels.get(name, name),
            )
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_title(self.title)
        axes.legend()
        return [fig]
