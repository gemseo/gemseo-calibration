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
"""Oscillator to generate signals."""

from __future__ import annotations

from math import exp
from typing import TYPE_CHECKING
from typing import Callable

from gemseo.core.chains.chain import MDOChain
from gemseo.disciplines.auto_py import AutoPyDiscipline

from gemseo_calibration.signal.ode_signal_generator import ODESignalGenerator

if TYPE_CHECKING:
    from gemseo_calibration.signal.base_signal_generator import Data


def compute_oscillator_rhs(
    time: float = 0.0,
    position: float = 1.0,
    velocity: float = 0.0,
    omega: float = 1.0,
) -> tuple[float, float]:
    """Compute the RHS of the ODE of a harmonic oscillator.

    Args:
        time: The time value.
        position: The position of the oscillator at ``time``.
        velocity: The velocity of the oscillator at ``time``.
        omega: The angular velocity of the oscillator at ``time``.

    Returns:
        The time derivative of the position of the oscillator,
        then the time derivative of the velocity of the oscillator.
    """
    position_dot = velocity
    velocity_dot = -(omega**2) * position
    return position_dot, velocity_dot


def compute_omega_rhs(time: float = 0.0, omega: float = 1.0, a: float = 2e-2) -> float:
    r"""Compute the RHS of the ODE of an angular velocity decreasing exponentially.

    .. math::

       \frac{\partial\omega}{\partial t} = -a\exp(-at)

    with :math:`a>0`.

    Args:
        time: The time value.
        omega: The angular velocity at ``time``.
        a: The positive rate at ``time``.

    Returns:
        The time derivative of the angular velocity of the oscillator.
    """
    omega_dot = -a * exp(-a * time)
    return omega_dot  # noqa: RET504


class Oscillator(ODESignalGenerator):
    """An oscillator to generate signals."""

    def __init__(self, omega: Data | Callable[[Data, Data], Data] = 0.0) -> None:
        """
        Args:
            omega: The angular velocity of the oscillator.
                If this velocity is time-independent,
                ``omega`` must be a number.
                If this velocity is time-dependent,
                ``omega`` must be
                a function of the form ``domega_dtime = f(time, omega)``
                to compute the RHS of the ODE defining the angular velocity.
                If ``0``, use the RHS computing ``-a * exp(-a * time)`` with ``a=2e-2``.
        """  # noqa: D205 D212
        rhs_discipline = AutoPyDiscipline(compute_oscillator_rhs)
        state_names = ["omega", "position", "velocity"]
        if callable(omega):
            rhs_discipline = MDOChain([AutoPyDiscipline(omega), rhs_discipline])
        elif omega == 0.0:
            rhs_discipline = MDOChain([
                AutoPyDiscipline(compute_omega_rhs),
                rhs_discipline,
            ])
        else:
            rhs_discipline.default_input_data["omega"] = omega
            state_names.remove("omega")

        super().__init__(rhs_discipline, state_names)
