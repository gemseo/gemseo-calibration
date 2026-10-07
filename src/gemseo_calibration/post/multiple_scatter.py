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

from typing import TYPE_CHECKING
from typing import ClassVar

from gemseo.post.dataset.base import BaseDatasetPlot

from gemseo_calibration.post.multiple_scatter_settings import MultipleScatter_Settings

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

    from gemseo.util.typing import RealArray


class MultipleScatter(BaseDatasetPlot[MultipleScatter_Settings]):
    """Overlay several scatter y_i versus x."""

    settings_class: ClassVar[type[MultipleScatter_Settings]] = MultipleScatter_Settings

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
        settings = self.settings
        y = settings.y
        if isinstance(y, str):
            y = [y]

        y_comp = {name: settings.y_comp.get(name, 0) for name in y}
        reference = self.dataset.get_view(variable_names=settings.x).to_numpy()[
            :, settings.x_comp
        ]
        settings.n_items = len(y)
        bounds = (min(reference), max(reference))
        return bounds, reference, y, y_comp
