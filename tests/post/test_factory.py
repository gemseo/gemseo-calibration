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
"""Test the :class:`.CalibrationPostFactory`."""
from __future__ import annotations

from pathlib import Path

import pytest
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from gemseo_calibration.post.factory import CalibrationPostFactory

DATA = Path(__file__).parent / ".." / "data"


@pytest.fixture
def post_factory(monkeypatch):  # type: (...) -> CalibrationPostFactory
    """The factory of post-processors dedicated to calibration."""
    monkeypatch.setenv("GEMSEO_PATH", DATA)
    return CalibrationPostFactory()


def test_init(post_factory):
    """Check that the factory is correctly initialized."""
    assert (
        post_factory.factory._Factory__base_class.__name__ == "CalibrationPostProcessor"
    )


def test_create(post_factory):
    """Check that a post-processor is correctly created."""
    post = post_factory.create(Rosenbrock(), 1, 2, 3, "NewCalibrationPostProcessor")
    assert post.__class__.__name__ == "NewCalibrationPostProcessor"
    assert post._reference_data == 1
    assert post._prior_model_data == 2
    assert post._posterior_model_data == 3


def test_execute(post_factory):
    """Check that a post-processor is correctly executed.

    Args:
        factory (CalibrationPostFactory): A factory
            to post-process calibration scenarios.
    """
    post = post_factory.execute(Rosenbrock(), 1, 2, 3, "NewCalibrationPostProcessor")
    assert post.executed


def test_posts(post_factory):
    """Check that a post-processor is correctly executed."""
    assert "NewCalibrationPostProcessor" in post_factory.posts
    assert "OptPostProcessor" not in post_factory.posts
