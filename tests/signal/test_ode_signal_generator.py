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
from __future__ import annotations

import pytest
from gemseo.disciplines.auto_py import AutoPyDiscipline
from numpy import array
from numpy import linspace
from numpy.testing import assert_almost_equal

from gemseo_calibration.problems.signal.oscillator import compute_oscillator_rhs
from gemseo_calibration.signal.ode_signal_generator import ODESignalGenerator


@pytest.fixture(scope="module")
def rhs_discipline() -> AutoPyDiscipline:
    """A discipline computing the RHS of an ODE."""
    return AutoPyDiscipline(compute_oscillator_rhs)


def test_rhs_discipline(rhs_discipline):
    """Check the property rhs_discipline."""
    generator = ODESignalGenerator(rhs_discipline, ("position", "velocity"))
    assert generator.rhs_discipline == rhs_discipline
    with pytest.raises(AttributeError):
        generator.rhs_discipline = rhs_discipline


def f(t=0, x=0):
    dxdt = 1
    return dxdt  # noqa: RET504


def test_time_name():
    """Check that ODESignalGenerator can use a custom time name."""
    generator = ODESignalGenerator(AutoPyDiscipline(f), ("x",), time_name="t")
    time = linspace(0.0, 1.0, num=3)
    signal = generator.generate(time, {"x": array([0.0])})
    assert_almost_equal(signal.final["x"], array([1.0]))


# See tests/problems/signal for the other tests.
