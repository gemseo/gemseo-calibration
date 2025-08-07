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
"""Time series based on a ODE discipline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.disciplines.ode.ode_discipline import ODEDiscipline
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT

from gemseo_calibration.signal.base_signal_generator import BaseSignalGenerator
from gemseo_calibration.signal.base_signal_generator import Data
from gemseo_calibration.signal.base_signal_generator import Signal

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

    from gemseo.core.discipline.discipline import Discipline
    from gemseo.typing import RealArray


class ODESignalGenerator(BaseSignalGenerator):
    """A signal generator based on a ODE discipline."""

    __rhs_discipline: Discipline
    """The RHS discipline."""

    __state_names: Iterable[str]
    """The names of the state variables."""

    __time_name: str
    """The name of the time variable."""

    def __init__(
        self,
        rhs_discipline: Discipline,
        state_names: Iterable[str],
        time_name: str = "time",
    ) -> None:
        """
        Args:
            rhs_discipline: The discipline defining the right-hand side (RHS)
                of the ordinary differential equation (ODE).
            state_names: The names of the state variables.
            time_name: The name of the time variable.
        """  # noqa: D205 D212
        super().__init__()
        self.__rhs_discipline = rhs_discipline
        self.__state_names = state_names
        self.__time_name = time_name
        self.grammar = rhs_discipline.io.input_grammar.__class__("Variables")
        self.grammar.update(rhs_discipline.io.input_grammar)
        self.grammar.update(rhs_discipline.io.output_grammar)

    @property
    def rhs_discipline(self) -> Discipline:
        """The discipline defining the RHS of the ODE."""
        return self.__rhs_discipline

    def generate(  # noqa: D102
        self,
        times: RealArray,
        initial_state_values: Mapping[str, Data],
        parameter_values: Mapping[str, Data] = READ_ONLY_EMPTY_DICT,
    ) -> Signal:
        ode_discipline = ODEDiscipline(
            self.__rhs_discipline,
            times,
            state_names=self.__state_names,
            time_name=self.__time_name,
            return_trajectories=True,
        )
        ode_discipline.default_input_data.update({
            f"initial_{k}": v for k, v in initial_state_values.items()
        })
        ode_discipline.default_input_data.update(parameter_values)
        ode_discipline.execute()
        data = ode_discipline.io.data
        return Signal(
            evolution={name: data[name] for name in initial_state_values},
            final={name: data[f"final_{name}"] for name in initial_state_values},
            times=times,
        )
