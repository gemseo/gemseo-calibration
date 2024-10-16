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
"""A discipline evaluating the quality of another one with respect to reference data."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.custom_doe.custom_doe import CustomDOE
from gemseo.core.discipline.discipline import Discipline
from gemseo.core.grammars.json_grammar import JSONGrammar
from gemseo.disciplines.scenario_adapters.mdo_scenario_adapter import MDOScenarioAdapter
from gemseo.scenarios.doe_scenario import DOEScenario
from numpy import array
from numpy import hstack

from gemseo_calibration.measures.factory import CalibrationMeasureFactory

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from gemseo.scenarios.scenario import Scenario

    from gemseo_calibration.measure import CalibrationMeasure as CalibrationMeasure_
    from gemseo_calibration.measure import DataType


class CalibrationMeasure(NamedTuple):
    """A calibration measure."""

    output: str
    """The name of the output."""

    measure: str = "MSE"
    """The name of the measure to compare the observed and simulated outputs."""

    mesh: str | None = None
    """The name of the irregular mesh associated with the output if any.

    To be used when the output is a 1D function discretized over an irregular mesh.
    """

    weight: float | None = None
    """The weight of this measure when the measure is an element of a collection.

    The weight must be between 0 and 1. The sum of the weights of the elements in the
    collection must be 1.
    """


class Calibrator(MDOScenarioAdapter):
    """A discipline with parameters calibrated from reference input-output data.

    When it is executed from parameters values, it computes the calibration measure with
    respect to the reference data, provided through the
    [set_reference_data][gemseo_calibration.calibrator.Calibrator.set_reference_data]
    method.
    """

    __ALGO_OPTIONS = "algo_options"

    def __init__(
        self,
        disciplines: Discipline | list[Discipline],
        input_names: str | Iterable[str],
        control_outputs: CalibrationMeasure | Sequence[CalibrationMeasure],
        parameter_names: str | Iterable[str],
        formulation: str = "MDF",
        **formulation_options: Any,
    ) -> None:
        """
        Args:
            disciplines: The disciplines
                whose parameters must be calibrated from the reference data.
            input_names: The names of the inputs to be considered for the calibration.
            control_outputs: The names of the outputs used to calibrate the disciplines
                with the name of the calibration measure and the corresponding weight
                comprised between 0 and 1 (the weights must sum to 1).
                When the output is a 1D function discretized over an irregular mesh,
                the name of the mesh can be provided.
                E.g. `CalibrationMeasure(output="z", measure="MSE")`
                `CalibrationMeasure(output="z", measure="MSE", weight=0.3)`
                or `CalibrationMeasure(output="z", measure="MSE", mesh="z_mesh")`
                Lastly, `CalibrationMeasure` can be imported
                from [gemseo-calibration.calibrator][gemseo-calibration.calibrator].
            parameter_names: The names of the parameters to be calibrated.
            formulation: The name of a formulation
                to manage the multidisciplinary coupling.
            **formulation_options: The options of the formulation.
        """  # noqa: D205,D212,D415
        self.__measure_factory = CalibrationMeasureFactory()
        input_names = self.__to_iterable(input_names, str)
        control_outputs = self.__to_iterable(control_outputs, CalibrationMeasure)
        parameter_names = self.__to_iterable(parameter_names, str)
        disciplines = self.__to_iterable(disciplines, Discipline)
        control_output = control_outputs[0]
        objective_name = control_output.output
        mesh_name = control_output.mesh
        input_space = DesignSpace()
        for name in input_names:
            input_space.add_variable(name)

        doe_scenario = DOEScenario(
            disciplines,
            formulation,
            objective_name,
            input_space,
            **formulation_options,
        )
        if mesh_name:
            doe_scenario.add_observable(mesh_name)

        self.__add_observables(control_outputs, doe_scenario)

        doe_scenario.default_input_data = {
            "algo": CustomDOE.__name__,
            self.__ALGO_OPTIONS: {"samples": None},
        }

        self.__names_to_measures = {}
        self.__measures = []
        self.objective_name, output_names = self._add_measure(control_outputs)
        super().__init__(doe_scenario, parameter_names, output_names, name="Calibrator")
        self.__update_output_grammar()
        self.__reference_data = {}

    @staticmethod
    def __add_observables(
        calibration_measures: Iterable[CalibrationMeasure], scenario: Scenario
    ) -> None:
        """Add observables to a scenario.

        Args:
            calibration_measures: The calibration measures to be added as observables.
            scenario: The scenario.
        """
        for calibration_measure in calibration_measures:
            output_name = calibration_measure.output
            mesh_name = calibration_measure.mesh
            scenario.add_observable(output_name)
            if mesh_name:
                scenario.add_observable(mesh_name)

    @staticmethod
    def __to_iterable(obj: Any, cls: type) -> Iterable[Any]:
        """Cast an object to an iterable.

        Args:
            obj: The object to cast.
            cls: The class of the elements of the iterable.

        Returns:
            An iterable of objects.
        """
        if isinstance(obj, cls):
            return [obj]
        return obj

    def _reset_optimization_problem(self) -> None:
        self.scenario.formulation.optimization_problem.reset()

    def __update_output_grammar(self) -> None:
        """Redefine the output grammar from the names of the output measures.

        E.g. MSE(y,z) is the name of the MSE measure applied to the outputs y and z.
        """
        output_grammar = JSONGrammar("outputs")
        output_grammar.update_from_names(self.__names_to_measures.keys())
        self.output_grammar = output_grammar

    def set_reference_data(self, reference_data: DataType) -> None:
        """Pass the reference data to the scenario and to the measures.

        Args:
            reference_data: The reference data with which to compare the discipline.
        """
        self.__reference_data = reference_data
        design_space = self.scenario.design_space
        for name in tuple(design_space):
            design_space.remove_variable(name)
            design_space.add_variable(name, size=reference_data[name].shape[1])

        self.scenario.default_input_data[self.__ALGO_OPTIONS]["samples"] = hstack([
            reference_data[name] for name in self.scenario.get_optim_variable_names()
        ])
        for measure in self.__measures:
            measure.set_reference_data(self.__reference_data)

    def _run(self) -> None:
        root_logger = logging.getLogger()
        saved_level = root_logger.level
        root_logger.setLevel(logging.WARNING)
        super()._run()
        root_logger.setLevel(saved_level)

    def _post_run(self) -> None:
        model_dataset = self.scenario.to_dataset().to_dict_of_arrays(False)
        for name, measure in self.__names_to_measures.items():
            self.io.data[name] = array([measure.func(model_dataset)])

    @property
    def maximize_objective_measure(self) -> bool:
        """Whether to maximize the calibration measure related to the objectives."""
        return self.__names_to_measures[self.objective_name].maximize

    def _add_measure(
        self,
        control_outputs: CalibrationMeasure | Iterable[CalibrationMeasure],
    ) -> tuple[str, list[str]]:
        """Create a new calibration measure and add it to the outputs of the adapter.

        The purpose of this method is to decouple adding a measure from updating
        the output grammar because during the call from __init__ the grammar instances
        are not yet existing before calling super().__init__.

        Args:
            control_outputs: The names of the outputs used to calibrate the disciplines
                with the name of the calibration measure and the corresponding weight
                comprised between 0 and 1 (the weights must sum to 1).
                When the output is a 1D function discretized over an irregular mesh,
                the name of the mesh can be provided.
                E.g. `CalibrationMeasure(output="z", measure="MSE")`
                `CalibrationMeasure(output="z", measure="MSE", weight=0.3)`
                or `CalibrationMeasure(output="z", measure="MSE", mesh="z_mesh")`
                Lastly, `CalibrationMeasure` can be imported
                from [gemseo-calibration.calibrator][gemseo-calibration.calibrator].

        Returns:
            The name of the calibration measure applied to the outputs.
        """
        control_outputs = self.__update_weights(control_outputs)
        name = ""
        control_outputs = self.__to_control_outputs(control_outputs)
        control_output = control_outputs[0]
        measure, output_names = self.__create_measure(control_output)
        self.__measures.append(measure)
        maximize = measure.maximize
        weight = control_output.weight
        if weight != 1.0:
            name += f"{weight}*{control_output.measure}[{measure.full_output_name}]"
            measure = measure * weight
        else:
            name += f"{control_output.measure}[{measure.full_output_name}]"
        for control_output in control_outputs[1:]:
            measure_, output_names_ = self.__create_measure(control_output)
            self.__measures.append(measure_)
            if measure_.maximize == maximize:
                weight = control_output.weight
            else:
                weight = -control_output.weight
            measure = measure + measure_ * weight
            output_names.extend(output_names_)
            name += f"+{weight}*{control_output.measure}[{measure_.full_output_name}]"

        measure.maximize = maximize
        measure.name = name
        self.__names_to_measures[name] = measure
        return name, list(set(output_names))

    @staticmethod
    def __to_control_outputs(
        control_outputs: CalibrationMeasure | Iterable[CalibrationMeasure],
    ) -> Iterable[CalibrationMeasure]:
        """Force control output(s) to be an iterator of calibration measures.

        Args:
            control_outputs: The control output(s).
        """
        if isinstance(control_outputs, CalibrationMeasure):
            control_outputs = [control_outputs]
        return control_outputs

    def add_measure(
        self,
        control_outputs: CalibrationMeasure | Iterable[CalibrationMeasure],
    ) -> tuple[str, list[str]]:
        """Create a new calibration measure and add it to the outputs of the adapter.

        Args:
            control_outputs: The names of the outputs used to calibrate the disciplines
                with the name of the calibration measure and the corresponding weight
                comprised between 0 and 1 (the weights must sum to 1).
                When the output is a 1D function discretized over an irregular mesh,
                the name of the mesh can be provided.
                E.g. ``CalibrationMeasure(output="z", measure="MSE")``
                ``CalibrationMeasure(output="z", measure="MSE", weight=0.3)``
                or ``CalibrationMeasure(output="z", measure="MSE", mesh="z_mesh")``
                Lastly, ``CalibrationMeasure`` can be imported
                from :mod:`gemseo-calibration.scenario`.

        Returns:
            The name of the calibration measure applied to the outputs.
        """
        control_outputs = self.__to_control_outputs(control_outputs)
        self.__add_observables(control_outputs, self.scenario)
        return_values = self._add_measure(control_outputs)
        self.__update_output_grammar()
        return return_values

    def __update_weights(
        self, control_outputs: Sequence[CalibrationMeasure]
    ) -> Sequence[CalibrationMeasure]:
        """Update the weights of the control outputs.

        Args:
            control_outputs: The control outputs.

        Returns:
            The updated control outputs.

        Raises:
            ValueError: When a weight is outside [0, 1]
                or when the weights do not sum to 1.
        """
        total_weight = 0
        missing_weight_indices = []
        for index, control_output in enumerate(control_outputs):
            weight = control_output.weight
            if weight is None:
                missing_weight_indices.append(index)
                continue

            if not 0 < weight < 1:
                msg = "The weight must be comprised between 0 and 1."
                raise ValueError(msg)

            total_weight += control_output.weight

        if not missing_weight_indices:
            if total_weight != 1:
                msg = "The weights must sum to 1."
                raise ValueError(msg)
            return control_outputs

        if total_weight >= 1:
            msg = "The weights must sum to 1."
            raise ValueError(msg)

        missing_weight = (1 - total_weight) / len(missing_weight_indices)
        for index in missing_weight_indices:
            control_output = control_outputs[index]
            control_outputs[index] = CalibrationMeasure(
                output=control_output.output,
                measure=control_output.measure,
                mesh=control_output.mesh,
                weight=missing_weight,
            )

        return control_outputs

    def __create_measure(
        self, control_output: CalibrationMeasure
    ) -> tuple[CalibrationMeasure_, list[str]]:
        if control_output.mesh:
            measure = self.__measure_factory.create(
                control_output.measure,
                output_name=control_output.output,
                mesh_name=control_output.mesh,
            )
            return measure, [control_output.output, control_output.mesh]

        measure = self.__measure_factory.create(
            control_output.measure, output_name=control_output.output
        )
        return measure, [control_output.output]

    @property
    def reference_data(self) -> DataType:
        """The reference data used for the calibration."""
        return self.__reference_data
