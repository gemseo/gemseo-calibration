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
import pytest
from gemseo.typing import RealArray
from numpy import array
from numpy import linspace
from numpy.testing import assert_almost_equal

from gemseo_calibration.problems.signal.oscillator import Oscillator
from gemseo_calibration.signal.signal_generator_discipline import (
    SignalGeneratorDiscipline,
)


@pytest.fixture(scope="module")
def oscillator() -> Oscillator:
    """An oscillator."""
    return Oscillator()


@pytest.fixture(scope="module")
def times() -> RealArray:
    """The times of interest."""
    return linspace(0.0, 1.0, num=5)


def test_basic(oscillator, times):
    """Check basic use."""
    discipline = SignalGeneratorDiscipline(
        oscillator,
        ("position", "velocity", "omega"),
        (),
        ("position",),
        times,
    )
    assert tuple(discipline.io.input_grammar.names) == (
        "initial_position",
        "initial_velocity",
        "initial_omega",
    )
    assert tuple(discipline.io.output_grammar.names) == ("position",)
    discipline.execute({
        "initial_position": 1.0,
        "initial_velocity": 0.0,
        "initial_omega": 1.5,
    })
    assert_almost_equal(
        discipline.io.data["position"],
        array([1.0, 0.9306871, 0.7327731, 0.4343574, 0.0766517]),
        decimal=2,
    )


def test_parameter_names(oscillator, times):
    """Check the use of parameter names."""
    discipline = SignalGeneratorDiscipline(
        oscillator,
        ("position", "velocity", "omega"),
        ("a",),
        ("position",),
        times,
    )
    assert tuple(discipline.io.input_grammar.names) == (
        "a",
        "initial_position",
        "initial_velocity",
        "initial_omega",
    )
    assert tuple(discipline.io.output_grammar.names) == ("position",)
    discipline.execute({
        "initial_position": 1.0,
        "initial_velocity": 0.0,
        "initial_omega": 1.5,
        "a": 1e-2,
    })
    assert_almost_equal(
        discipline.io.data["position"],
        array([1.0, 0.930612, 0.7322193, 0.4327589, 0.0736661]),
        decimal=2,
    )


def test_not_all_state_names(oscillator, times):
    """Check the use of certain state names."""
    discipline = SignalGeneratorDiscipline(
        oscillator,
        ("position", "velocity"),
        (),
        ("position",),
        times,
    )
    assert tuple(discipline.io.input_grammar.names) == (
        "initial_position",
        "initial_velocity",
    )
    assert tuple(discipline.io.output_grammar.names) == ("position",)
    discipline.execute({"initial_position": 1.0, "initial_velocity": 0.0})
    assert_almost_equal(
        discipline.io.data["position"],
        array([1.0, 0.9689715, 0.8780239, 0.7329289, 0.5429478]),
        decimal=2,
    )
