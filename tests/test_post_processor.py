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
"""Test the class CalibrationPostProcessor."""

from __future__ import annotations

from gemseo.post.opt_post_processor import OptPostProcessor
from gemseo.post.opt_post_processor import OptPostProcessorOptionType
from gemseo.problems.analytical.rosenbrock import Rosenbrock

from gemseo_calibration.post_processor import CalibrationPostProcessor


class NewCalibrationPostProcessor(CalibrationPostProcessor):
    """A new calibration post processor."""

    def _plot(self, **options: OptPostProcessorOptionType) -> None:
        return


def test_post():
    """Test that the base class CalibrationPostProcessor is correctly initialized."""
    opt_problem = Rosenbrock()
    post = NewCalibrationPostProcessor(opt_problem, 1, 2, 3)
    assert post.opt_problem == opt_problem
    assert isinstance(post, OptPostProcessor)
    assert post._reference_data == 1
    assert post._prior_model_data == 2
    assert post._posterior_model_data == 3
