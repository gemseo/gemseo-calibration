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
"""A discipline wrapping a time series."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Union

from gemseo.core.discipline.discipline import Discipline
from gemseo.utils.pydantic_ndarray import NDArrayPydantic

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo import StrKeyMapping
    from gemseo.typing import RealArray

    from gemseo_calibration.signal.base_signal_generator import BaseSignalGenerator


class SignalGeneratorDiscipline(Discipline):
    """A discipline wrapping a time series."""

    __parameter_names: set[str]
    """The names of the parameters."""

    __signal_generator: BaseSignalGenerator
    """The signal generator."""

    __times: RealArray
    """The times of interest in ascending order, from initial time to final time."""

    default_grammar_type = Discipline.GrammarType.PYDANTIC

    def __init__(
        self,
        signal_generator: BaseSignalGenerator,
        state_names: Iterable[str],
        parameter_names: Iterable[str],
        output_names: Iterable[str],
        times: RealArray,
    ) -> None:
        """
        Args:
            signal_generator: The signal generator.
            state_names: The names of the variables in the signal generator
                used as discipline input variables.
            parameter_names: The names of the parameters in the signal generator
                used as discipline input variables.
            output_names: The names of the variables in the signal generator
                used as discipline output variables.
            times: The times of interest in ascending order,
                from initial time to final time.
        """  # noqa: D205 D212
        super().__init__()
        input_names = (*(f"initial_{k}" for k in state_names), *parameter_names)
        self.io.input_grammar.update_from_types(
            dict.fromkeys(input_names, Union[float, NDArrayPydantic])
        )
        self.io.output_grammar.update_from_types(
            dict.fromkeys(output_names, Union[float, NDArrayPydantic])
        )
        self.__parameter_names = set(parameter_names)
        self.__signal_generator = signal_generator
        self.__times = times

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        parameter_values = {k: input_data[k] for k in self.__parameter_names}
        input_data = {
            k[8:]: v for k, v in input_data.items() if k not in self.__parameter_names
        }
        output_data = self.__signal_generator.generate(
            self.__times, input_data, parameter_values=parameter_values
        ).evolution
        return {k: output_data[k] for k in self.io.output_grammar}
