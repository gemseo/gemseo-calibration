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


def test_init(post_factory):
    """Check that the factory is correctly initialized."""
    assert post_factory._CLASS.__name__ == "CalibrationPostProcessor"


def test_posts(post_factory):
    """Check that a post-processor is correctly executed."""
    assert "DataVersusModel" in post_factory.class_names
    assert "OptHistoryView" not in post_factory.class_names
