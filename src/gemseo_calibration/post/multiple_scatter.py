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
r"""Overlay several scatter plots from a `Dataset`.

A scatter plot represents a set of points $\{x_i,y_i\}_{1\leq i \leq n}$ as markers on a
classical plot, while a multiple-scatter plot represents a set of points
$\{x_i,y_{i,1},\ldots,y_{i,d}\}_{1\leq i \leq n}$ as markers on a classical plot, with
one color per series $\{y_i\}_{1\leq i \leq n}$.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from gemseo.post.dataset.dataset_plot import DatasetPlot

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

    from gemseo.datasets.dataset import Dataset
    from gemseo.typing import RealArray


class MultipleScatter(DatasetPlot):
    """Overlay several scatter y_i versus x."""

    def __init__(
        self,
        dataset: Dataset,
        x: str,
        y: str | Iterable[str],
        x_comp: int = 0,
        y_comp: Mapping[str, int] = MappingProxyType({}),
    ) -> None:
        """
        Args:
            x: The name of the variable on the x-axis.
            y: The names of the variables on the y-axis.
            x_comp: The component of x.
            y_comp: The components of y,
                where the names are the names of the variables
                and the values are the components.
                If empty or if a name is missing,
                use the first component.
        """  # noqa: D205 D212 D415
        super().__init__(dataset=dataset, x=x, y=y, x_comp=x_comp, y_comp=y_comp)

    def _create_specific_data_from_dataset(
        self,
    ) -> tuple[tuple[float, float], RealArray, Iterable[str], Mapping[str, int]]:
        """
        Returns:
            The lower and upper bounds of the reference values,
            the reference values,
            the names of the variables on the y-axis and
            the components of these variables.
        """  # noqa: D205, D212, D415
        x = self._specific_settings.x
        y = self._specific_settings.y
        if isinstance(y, str):
            y = [y]

        y_comp = self._specific_settings.y_comp or {}
        for name in y:
            y_comp[name] = y_comp.get(name, 0)

        reference = self.dataset.get_view(variable_names=x).to_numpy()[
            :, self._specific_settings.x_comp
        ]
        self._n_items = len(y)
        bounds = (min(reference), max(reference))
        return bounds, reference, y, y_comp
