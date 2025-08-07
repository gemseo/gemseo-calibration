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
"""Base class for signal generators."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import Union

from gemseo.typing import RealArray
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gemseo.core.grammars.base_grammar import BaseGrammar


Data = Union[float, RealArray]


class Signal(NamedTuple):
    """A multivariate signal."""

    evolution: dict[str, RealArray]
    """The values of the state variables at time of interest."""

    final: dict[str, Data]
    """The values of the state variables at final time."""

    times: RealArray
    """The times of interest in ascending order, from initial time to final time."""


class BaseSignalGenerator(metaclass=ABCGoogleDocstringInheritanceMeta):
    """The base class for signal generators."""

    grammar: BaseGrammar
    """The grammar of variables."""

    @abstractmethod
    def generate(  # noqa: D102
        self,
        times: RealArray,
        initial_state_values: Mapping[str, Data],
        parameter_values: Mapping[str, Data] = READ_ONLY_EMPTY_DICT,
    ) -> Signal:
        """Generate a multivariate signal.

        Args:
            times: The times of interest in ascending order,
                from initial time to final time.
            initial_state_values: The values of the state variables
                at initial time.
            parameter_values: The values of the parameters.

        Returns:
            A multivariate signal.
        """
