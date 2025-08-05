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

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.custom_doe.custom_doe import CustomDOE
from gemseo.core.discipline.discipline import Discipline
from gemseo.core.grammars.json_grammar import JSONGrammar
from gemseo.disciplines.scenario_adapters.mdo_scenario_adapter import MDOScenarioAdapter
from gemseo.scenarios.doe_scenario import DOEScenario
from numpy import array
from numpy import hstack

from gemseo_calibration.metrics.factory import CalibrationMetricFactory
from gemseo_calibration.metrics.settings import CalibrationMetricSettings

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from gemseo.formulations.base_formulation_settings import BaseFormulationSettings

    from gemseo_calibration.metrics.base_calibration_metric import BaseCalibrationMetric
    from gemseo_calibration.metrics.base_calibration_metric import DataType


class Calibrator(MDOScenarioAdapter):
    """A discipline with parameters calibrated from reference input-output data.

    When it is executed from parameters values, it computes the calibration metric with
    respect to the reference data, provided through the
    [set_reference_data][gemseo_calibration.calibrator.Calibrator.set_reference_data]
    method.
    """

    def __init__(
        self,
        disciplines: Discipline | list[Discipline],
        input_names: str | Iterable[str],
        metric_settings_models: CalibrationMetricSettings
        | Sequence[CalibrationMetricSettings],
        parameter_names: str | Iterable[str],
        formulation_settings_model: BaseFormulationSettings | None = None,
        **formulation_settings: Any,
    ) -> None:
        """
        Args:
            disciplines: The disciplines
                whose parameters must be calibrated from the reference data.
            input_names: The names of the inputs to be considered for the calibration.
            metric_settings_models: A collection of calibration settings,
                including the name of the observed output,
                the name of the calibration metric
                and the corresponding weight comprised between 0 and 1
                (the weights must sum to 1).
                When the output is a 1D function discretized over a mesh,
                the name of the mesh can be provided.
                E.g. `CalibrationMetricSettings(output_name="z", metric_name="MSE")`
                `CalibrationMetricSettings(output_name="z", metric_name="MSE", weight=0.3)`
                or
                `CalibrationMetricSettings(output_name="z", metric_name="MSE", mesh_name="z_mesh")`
                Lastly, `CalibrationMetric` can be imported
                from [gemseo_calibration.calibrator][gemseo_calibration.calibrator].
            parameter_names: The names of the parameters to be calibrated.
            formulation_settings_model: The MDO formulation settings
                as a Pydantic model.
                If ``None``, use ``**settings``.
            **formulation_settings: The MDO formulation settings,
                including the formulation name (use the keyword ``"formulation_name"``).
                These arguments are ignored when ``settings_model`` is not ``None``.
                If none and ``settings_model`` is ``None``,
                the calibrator uses the default MDF formulation.
        """  # noqa: D205,D212,D415,E501
        self.__metric_factory = CalibrationMetricFactory()
        input_names = self.__to_iterable(input_names, str)
        metric_settings_models = self.__to_iterable(
            metric_settings_models, CalibrationMetricSettings
        )
        parameter_names = self.__to_iterable(parameter_names, str)
        disciplines = self.__to_iterable(disciplines, Discipline)
        input_space = DesignSpace()
        for input_name in input_names:
            input_space.add_variable(input_name)

        if (
            formulation_settings_model is None
            and "formulation_name" not in formulation_settings
        ):
            formulation_settings["formulation_name"] = "MDF"

        doe_scenario = DOEScenario(
            disciplines,
            metric_settings_models[0].output_name,
            input_space,
            formulation_settings_model=formulation_settings_model,
            **formulation_settings,
        )
        function_names = [metric_settings_models[0].output_name]
        for metric_settings_model in metric_settings_models:
            for name in (
                metric_settings_model.output_name,
                metric_settings_model.mesh_name,
            ):
                if name and name not in function_names:
                    doe_scenario.add_observable(name)
                    function_names.append(name)

        doe_scenario.set_algorithm(algo_name=CustomDOE.__name__)

        self.__names_to_metrics = {}
        self.__metrics = []
        self.objective_name, output_names = self._add_metric(metric_settings_models)
        super().__init__(doe_scenario, parameter_names, output_names, name="Calibrator")
        self.__update_output_grammar()
        self.__reference_data = {}

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
        """Redefine the output grammar from the names of the output metrics.

        E.g. MSE(y,z) is the name of the MSE metric applied to the outputs y and z.
        """
        output_grammar = JSONGrammar("outputs")
        output_grammar.update_from_names(self.__names_to_metrics.keys())
        self.output_grammar = output_grammar

    def set_reference_data(self, reference_data: DataType) -> None:
        """Pass the reference data to the scenario and to the metrics.

        Args:
            reference_data: The reference data with which to compare the discipline.
        """
        self.__reference_data = reference_data
        design_space = self.scenario.design_space
        for name in tuple(design_space):
            design_space.remove_variable(name)
            design_space.add_variable(name, size=reference_data[name].shape[1])

        self.scenario.set_algorithm(
            algo_name="CustomDOE",
            samples=hstack([
                reference_data[name]
                for name in self.scenario.get_optim_variable_names()
            ]),
        )
        for metric in self.__metrics:
            metric.set_reference_data(self.__reference_data)

    def _execute(self) -> None:
        root_logger = logging.getLogger()
        saved_level = root_logger.level
        root_logger.setLevel(logging.WARNING)
        super()._execute()
        root_logger.setLevel(saved_level)

    def _post_run(self) -> None:
        model_dataset = self.scenario.to_dataset().to_dict_of_arrays(False)
        for name, metric in self.__names_to_metrics.items():
            self.io.data[name] = array([metric.func(model_dataset)])

    @property
    def maximize_objective_metric(self) -> bool:
        """Whether to maximize the calibration metric related to the objectives."""
        return self.__names_to_metrics[self.objective_name].maximize

    def _add_metric(
        self,
        metric_settings_models: CalibrationMetricSettings
        | Iterable[CalibrationMetricSettings],
    ) -> tuple[str, list[str]]:
        """Create a new calibration metric and add it to the outputs of the adapter.

        The purpose of this method is to decouple adding a metric from updating
        the output grammar because during the call from __init__ the grammar instances
        are not yet existing before calling super().__init__.

        Args:
            metric_settings_models: A collection of calibration settings.,

        Returns:
            The name of the calibration metric applied to the outputs.
        """  # noqa: E501
        metric_settings_models = self.__update_weights(metric_settings_models)
        name = ""
        metric_settings_models = self.__to_metric_settings(metric_settings_models)
        metric_settings_model = metric_settings_models[0]
        metric, output_names = self.__create_metric(metric_settings_model)
        self.__metrics.append(metric)
        maximize = metric.maximize
        weight = metric_settings_model.weight
        if weight != 1.0:
            name += (
                f"{weight}*{metric_settings_model.metric_name}"
                f"[{metric.full_output_name}]"
            )
            metric = metric * weight
        else:
            name += f"{metric_settings_model.metric_name}[{metric.full_output_name}]"
        for metric_settings_model in metric_settings_models[1:]:
            metric_, output_names_ = self.__create_metric(metric_settings_model)
            self.__metrics.append(metric_)
            if metric_.maximize == maximize:
                weight = metric_settings_model.weight
            else:
                weight = -metric_settings_model.weight
            metric = metric + metric_ * weight
            output_names.extend(output_names_)
            name += (
                f"+{weight}*{metric_settings_model.metric_name}"
                f"[{metric_.full_output_name}]"
            )

        metric.maximize = maximize
        metric.name = name
        self.__names_to_metrics[name] = metric
        return name, list(set(output_names))

    @staticmethod
    def __to_metric_settings(
        metric_settings_models: CalibrationMetricSettings
        | Iterable[CalibrationMetricSettings],
    ) -> Iterable[CalibrationMetricSettings]:
        """Force calibration settings model(s) to be a collection.

        Args:
            metric_settings_models: Either a calibration settings model
                or a collection of calibration settings models.

        Returns:
            A collection of calibration settings models.
        """
        if isinstance(metric_settings_models, CalibrationMetricSettings):
            metric_settings_models = [metric_settings_models]
        return metric_settings_models

    def add_metric(
        self,
        metric_settings_models: CalibrationMetricSettings
        | Iterable[CalibrationMetricSettings],
    ) -> tuple[str, list[str]]:
        """Create a new calibration metric and add it to the outputs of the adapter.

        Args:
            metric_settings_models: A collection of calibration settings,
                including the name of the observed output,
                the name of the calibration metric
                and the corresponding weight comprised between 0 and 1
                (the weights must sum to 1).
                When the output is a 1D function discretized over a mesh,
                the name of the mesh can be provided.
                E.g. `CalibrationMetricSettings(output_name="z", metric_name="MSE")`
                `CalibrationMetricSettings(output_name="z", metric_name="MSE", weight=0.3)`
                or
                `CalibrationMetricSettings(output_name="z", metric_name="MSE", mesh_name="z_mesh")`
                from
                [gemseo_calibration.metrics.settings][gemseo_calibration.metrics.settings].

        Returns:
            The name of the calibration metric applied to the outputs.
        """  # noqa: E501
        metric_settings_models = self.__to_metric_settings(metric_settings_models)
        function_names = [
            function_.name
            for function_ in self.scenario.formulation.optimization_problem.functions
            if function_ is not None
        ]
        for metric_settings_model in metric_settings_models:
            for name in (
                metric_settings_model.output_name,
                metric_settings_model.mesh_name,
            ):
                if name and name not in function_names:
                    self.scenario.add_observable(name)
                    function_names.append(name)

        return_values = self._add_metric(metric_settings_models)
        self.__update_output_grammar()
        return return_values

    @staticmethod
    def __update_weights(
        metric_settings_models: Sequence[CalibrationMetricSettings],
    ) -> Sequence[CalibrationMetricSettings]:
        """Update the weights of a collection of calibration settings.

        Args:
            metric_settings_models: A collection of calibration settings.

        Returns:
            The updated collection of calibration settings.

        Raises:
            ValueError: When a weight is outside [0, 1]
                or when the weights do not sum to 1.
        """
        total_weight = 0
        missing_weight_indices = []
        for index, metric_settings_model in enumerate(metric_settings_models):
            weight = metric_settings_model.weight
            if weight is None:
                missing_weight_indices.append(index)
                continue

            if not 0 < weight < 1:
                msg = "The weight must be comprised between 0 and 1."
                raise ValueError(msg)

            total_weight += metric_settings_model.weight

        if not missing_weight_indices:
            if total_weight != 1:
                msg = "The weights must sum to 1."
                raise ValueError(msg)
            return metric_settings_models

        if total_weight >= 1:
            msg = "The weights must sum to 1."
            raise ValueError(msg)

        missing_weight = (1 - total_weight) / len(missing_weight_indices)
        for index in missing_weight_indices:
            metric_setting = metric_settings_models[index]
            metric_settings_models[index] = CalibrationMetricSettings(
                output_name=metric_setting.output_name,
                metric_name=metric_setting.metric_name,
                mesh_name=metric_setting.mesh_name,
                weight=missing_weight,
            )

        return metric_settings_models

    def __create_metric(
        self, metric_settings_model: CalibrationMetricSettings
    ) -> tuple[BaseCalibrationMetric, list[str]]:
        """Create a calibration metric from settings.

        Args:
            metric_settings_model: The calibration settings.

        Returns:
            The calibration metric and the associated output name.
        """
        if metric_settings_model.mesh_name:
            metric = self.__metric_factory.create(
                metric_settings_model.metric_name,
                output_name=metric_settings_model.output_name,
                mesh_name=metric_settings_model.mesh_name,
            )
            return metric, [
                metric_settings_model.output_name,
                metric_settings_model.mesh_name,
            ]

        metric = self.__metric_factory.create(
            metric_settings_model.metric_name,
            output_name=metric_settings_model.output_name,
        )
        return metric, [metric_settings_model.output_name]

    @property
    def reference_data(self) -> DataType:
        """The reference data used for the calibration."""
        return self.__reference_data
