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
"""A matplotlib-based implementation of `MultipleScatter`.."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.post.dataset.plots._matplotlib.plot import MatplotlibPlot
from gemseo.utils.compatibility.matplotlib import get_color_map
from numpy import linspace

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

    from gemseo.typing import RealArray
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class MultipleScatter(MatplotlibPlot):
    """Overlay several scatter y_i versus x."""

    def _create_figures(
        self,
        fig: Figure | None,
        axes: Axes | None,
        bounds: tuple[float, float],
        reference: RealArray,
        y: Iterable[str],
        y_comp: Mapping[str, int],
    ) -> list[Figure]:
        """
        Args:
            bounds: The lower and upper bounds of the reference values.
            reference: The reference values.
            y: The names of the variables on the y-axis.
            y_comp: The components of these variables.
        """  # noqa: D205 D212
        fig, axes = self._get_figure_and_axes(fig, axes)
        axes.plot(bounds, bounds, color="gray", linestyle="--", marker="o")
        n_items = len(y)
        color = self._common_settings.color
        color_map = get_color_map(self._common_settings.colormap)
        color = color or [color_map(c) for c in linspace(0, 1, n_items)]
        if isinstance(color, str):
            color = [color] * n_items
        for index, name in enumerate(y):
            axes.plot(
                reference,
                self._common_dataset.get_view(variable_names=name).to_numpy()[
                    :, y_comp[name]
                ],
                color=color[index],
                marker="o",
                linestyle="",
                label=self._common_settings.labels.get(name, name),
            )
        axes.grid()
        axes.set_xlabel(self._common_settings.xlabel)
        axes.set_ylabel(self._common_settings.ylabel)
        axes.set_title(self._common_settings.title)
        axes.legend()
        return [fig]
