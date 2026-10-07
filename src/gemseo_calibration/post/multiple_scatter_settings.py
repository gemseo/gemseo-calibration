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
"""The settings for `MultipleScatter`."""

from __future__ import annotations

from gemseo.post.dataset.base_cartesian_settings import BaseCartesianDatasetPlotSettings
from pydantic import Field
from pydantic import NonNegativeInt  # noqa: TC002


class MultipleScatter_Settings(BaseCartesianDatasetPlotSettings):  # noqa: N801
    """The settings for `MultipleScatter`."""

    x: str = Field(description="The name of the variable on the x-axis.")

    y: str | list[str] = Field(description="The names of the variables on the y-axis.")

    x_comp: NonNegativeInt = Field(default=0, description="The component of x.")

    y_comp: dict[str, NonNegativeInt] = Field(
        default_factory=dict,
        description="""The components of y,
where the names are the names of the variables
and the values are the components.
If empty or if a name is missing,
use the first component.""",
    )
