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
"""Test the class MultipleScatter plotting variable y_i versus a variable x."""

from __future__ import annotations

import pytest
from gemseo.datasets.dataset import Dataset
from gemseo.utils.testing.helpers import image_comparison
from matplotlib import pyplot as plt
from numpy import array

from gemseo_calibration.post.multiple_scatter import MultipleScatter


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    """A dataset containing 3 samples of variables x, y and z (dim(z)=2)."""
    sample1 = [0.0, 0.0, 1.0, 0.25]
    sample2 = [0.5, 0.5, 0.75, 0.75]
    sample3 = [1.0, 1.0, 0.25, 1.0]
    data_array = array([sample1, sample2, sample3])
    variable_names_to_n_components = {"x": 1, "y1": 1, "y2": 2}
    return Dataset.from_array(
        data_array,
        variable_names=["x", "y1", "y2"],
        variable_names_to_n_components=variable_names_to_n_components,
    )


# the test parameters, it maps a test name to the inputs and references outputs:
# - the kwargs to be passed to MultipleScatter._plot
# - the expected file names without extension to be compared
TEST_PARAMETERS = {
    "default": ({"x": "x", "y": "y1"}, {}, ["default"]),
    "with_color": (
        {"x": "x", "y": ["y1"]},
        {"color": "red"},
        ["with_color"],
    ),
    "with_2_outputs": (
        {"x": "x", "y": ["y1", "y2"]},
        {},
        ["with_2_outputs"],
    ),
    "with_2_outputs_and_color": (
        {"x": "x", "y": ["y1", "y2"]},
        {"color": "red"},
        ["with_2_outputs_and_color"],
    ),
    "with_2_outputs_and_colors": (
        {"x": "x", "y": ["y1", "y2"]},
        {"color": ["red", "blue"]},
        ["with_2_outputs_and_colors"],
    ),
    "with_2_outputs_components": (
        {"x": "x", "y": ["y1", "y2"], "y_comp": {"y2": 1}},
        {},
        ["with_2_outputs_components"],
    ),
}


@pytest.mark.parametrize(
    ("options", "properties", "baseline_images"),
    TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=TEST_PARAMETERS.keys(),
)
@pytest.mark.parametrize("fig_and_axes", [False, True])
@image_comparison(None)
def test_plot(
    options, properties, baseline_images, dataset, pyplot_close_all, fig_and_axes
):
    """Test images created by MultipleScatter._plot against references."""
    plot = MultipleScatter(dataset, **options)
    fig, axes = (
        (None, None) if not fig_and_axes else plt.subplots(figsize=plot.fig_size)
    )
    for name, value in properties.items():
        setattr(plot, name, value)

    plot.execute(save=False, show=False, fig=fig, ax=axes)
