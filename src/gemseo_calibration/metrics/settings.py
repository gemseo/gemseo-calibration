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
"""The settings of a calibration metric."""

from __future__ import annotations

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import PositiveFloat


class CalibrationMetricSettings(BaseModel):
    """The settings of a calibration metric."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    output_name: str = Field(description="The name of the output.")

    metric_name: str = Field(
        default="MSE",
        description="""The name of the metric
to compare the observed and simulated outputs.""",
    )

    mesh_name: str = Field(
        default="",
        description="""The name of the mesh associated with the output if any.

To be used when the output is a 1D function discretized over a mesh.""",
    )

    weight: PositiveFloat | None = Field(
        default=None,
        description="""The weight of this calibration metric
when this calibration metric is an element of a collection of calibration metrics.

The weight must be between 0 and 1.
The sum of the weights of the elements in the collection must be 1.
In a collection,
all the calibration metrics with `weight` set to `None` will have the same weight.""",
    )
