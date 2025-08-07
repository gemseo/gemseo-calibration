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
from gemseo_calibration.problems.signal.oscillator import compute_omega_rhs
from gemseo_calibration.problems.signal.oscillator import compute_oscillator_rhs


@pytest.fixture(scope="module")
def times() -> RealArray:
    """The times of interest."""
    return linspace(0.0, 1.0, num=3)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, (0.0, -1.0)),
        ({"time": 1.0, "position": 2.0, "velocity": 3.0, "omega": 4.0}, (3.0, -32.0)),
    ],
)
def test_compute_oscillator_rhs(kwargs, expected):
    """Test the compute_oscillator_rhs function."""
    assert compute_oscillator_rhs(**kwargs) == expected


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, -0.02),
        ({"time": 1.0, "omega": 4.0, "a": 1e-2}, -0.009900498337491681),
    ],
)
def test_compute_omega_rhs(kwargs, expected):
    """Test the compute_omega_rhs function."""
    assert compute_omega_rhs(**kwargs) == pytest.approx(expected)


def test_oscillator(times):
    """Test the Oscillator class."""
    oscillator = Oscillator()
    signal = oscillator.generate(
        times, {"position": 1.5, "velocity": 0.2, "omega": 1.5}
    )

    assert id(signal.times) == id(times)

    evolution = signal.evolution
    assert set(evolution) == {"position", "velocity", "omega"}
    assert_almost_equal(evolution["position"], array([1.5, 1.1901095, 0.2487466]))
    assert_almost_equal(evolution["velocity"], array([0.2, -1.377731, -2.2109119]))
    assert_almost_equal(evolution["omega"], array([1.5, 1.4900498, 1.4801987]))

    final = signal.final
    assert set(final) == {"position", "velocity", "omega"}
    assert_almost_equal(final["position"], array([0.2487466]))
    assert_almost_equal(final["velocity"], array([-2.2109119]))
    assert_almost_equal(final["omega"], array([1.4801987]))


def test_oscillator_constant_omega(times):
    """Test the Oscillator class with a constant omega."""
    oscillator = Oscillator(omega=1.5)
    signal = oscillator.generate(times, {"position": 1.5, "velocity": 0.2})

    assert id(signal.times) == id(times)

    evolution = signal.evolution
    assert set(evolution) == {"position", "velocity"}
    assert_almost_equal(evolution["position"], array([1.5, 1.188393, 0.2389624]))
    assert_almost_equal(evolution["velocity"], array([0.2, -1.3873755, -2.2301765]))

    final = signal.final
    assert set(final) == {"position", "velocity"}
    assert_almost_equal(final["position"], array([0.2389624]))
    assert_almost_equal(final["velocity"], array([-2.2301765]))


def compute_omega_rhs_2(time: float = 0.0, omega: float = 1.0) -> float:
    domega_dt = -0.2
    return domega_dt  # noqa: RET504


def test_oscillator_time_varying_omega(times):
    """Test the Oscillator class with a time-varying omega."""
    oscillator = Oscillator(omega=compute_omega_rhs_2)
    signal = oscillator.generate(
        times, {"position": 1.5, "velocity": 0.2, "omega": 1.5}
    )

    assert id(signal.times) == id(times)

    evolution = signal.evolution
    assert set(evolution) == {"position", "velocity", "omega"}
    assert_almost_equal(evolution["position"], array([1.5, 1.2054435, 0.3363505]))
    assert_almost_equal(evolution["velocity"], array([0.2, -1.2921358, -2.0330212]))
    assert_almost_equal(evolution["omega"], array([1.5, 1.4, 1.3]))

    final = signal.final
    assert set(final) == {"position", "velocity", "omega"}
    assert_almost_equal(final["position"], array([0.3363505]))
    assert_almost_equal(final["velocity"], array([-2.0330212]))
    assert_almost_equal(final["omega"], array([1.3]))
