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
"""A module to calibrate a multidisciplinary system from data."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.scenarios.mdo_scenario import MDOScenario

from gemseo_calibration.calibrator import CalibrationMeasure
from gemseo_calibration.calibrator import Calibrator
from gemseo_calibration.post.factory import CalibrationPostFactory

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from gemseo.algos.base_driver_settings import BaseDriverSettings
    from gemseo.algos.design_space import DesignSpace
    from gemseo.core.discipline.discipline import Discipline
    from gemseo.post.base_post import BasePost
    from gemseo.typing import RealArray
    from gemseo.typing import StrKeyMapping

    from gemseo_calibration.measure import DataType


class CalibrationScenario(MDOScenario):
    """A scenario to calibrate a multidisciplinary system from reference data.

    Set from parameters,
    this multidisciplinary system computes output data from input data.

    The reference input-output data are used to calibrate the parameters
    so that the model output data are close to the reference output data
    for some outputs of interest.
    This distance is evaluated with a
    [CalibrationMeasure][gemseo_calibration.measure.CalibrationMeasure]
    to compare the discipline outputs with the reference data.

    Warning:
        Just like inputs,
        the parameters should be defined in the input grammars of the disciplines.

    The parameters are calibrated with the method
    [execute][gemseo_calibration.scenario.CalibrationScenario.execute]
    from an optimizer and a reference
    [IODataset][gemseo.datasets.io_dataset.IODataset].
    """

    prior_model_data: dict[str, RealArray]
    """The model data before the calibration."""

    posterior_model_data: dict[str, RealArray]
    """The model data after the calibration."""

    reference_data: StrKeyMapping
    """The reference data, if defined."""

    def __init__(
        self,
        disciplines: Discipline | list[Discipline],
        input_names: str | Iterable[str],
        control_outputs: CalibrationMeasure | Sequence[CalibrationMeasure],
        calibration_space: DesignSpace,
        formulation: str = "MDF",
        name: str = "",
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
                from :mod:`gemseo-calibration.scenario`.
            calibration_space: The space of the parameters to be calibrated,
                whose current values are consider as a prior for calibration.
            formulation: The name of a formulation
                to manage the multidisciplinary coupling.
            name: A name for this calibration scenario.
                If empty, use the name of the class.
            **formulation_options: The options of the formulation.
        """  # noqa: D205,D212,D415
        self.__prior_parameters = calibration_space.get_current_value(as_dict=True)
        self.__posterior_parameters = {}
        self.prior_model_data = {}
        self.posterior_model_data = {}
        self.__reference_data = {}
        calibrator = Calibrator(
            disciplines,
            input_names,
            control_outputs,
            calibration_space.variable_names,
            formulation=formulation,
            **formulation_options,
        )
        super().__init__(
            [calibrator],
            calibrator.objective_name,
            calibration_space,
            name=name or self.__class__.__name__,
            formulation_name="DisciplinaryOpt",
            maximize_objective=calibrator.maximize_objective_measure,
        )
        self.__calibration_post_factory = CalibrationPostFactory()

    def set_algorithm(  # noqa:D102
        self,
        reference_data: StrKeyMapping,
        algo_settings_model: BaseDriverSettings | None = None,
        **algo_settings: Any,
    ) -> None:
        self.__reference_data = reference_data
        super().set_algorithm(algo_settings_model=algo_settings_model, **algo_settings)

    def _run(self) -> None:
        self.calibrator.set_reference_data(self.__reference_data)
        self.calibrator.execute()
        self.prior_model_data = self.calibrator.scenario.to_dataset().to_dict_of_arrays(
            False
        )
        super()._run()
        self.__posterior_parameters = self.design_space.convert_array_to_dict(
            self.optimization_result.x_opt
        )
        self.calibrator.default_input_data = self.posterior_parameters
        self.calibrator.execute()
        self.posterior_model_data = (
            self.calibrator.scenario.to_dataset().to_dict_of_arrays(False)
        )

    @property
    def calibrator(self) -> Calibrator:
        """The discipline computing calibration measures from the parameter values."""
        return self.formulation.disciplines[0]

    @property
    def prior_parameters(self) -> DataType:
        """The values of the parameters before the calibration stage."""
        return self.__prior_parameters

    @property
    def posterior_parameters(self) -> DataType:
        """The values of the parameters after the calibration stage."""
        return self.__posterior_parameters

    def add_constraint(
        self,
        control_outputs: CalibrationMeasure | Iterable[CalibrationMeasure],
        constraint_type: MDOFunction.ConstraintType = MDOFunction.ConstraintType.EQ,
        constraint_name: str = "",
        value: float = 0.0,
        positive: bool = False,
    ) -> None:
        """Define a constraint from a calibration measure related to discipline outputs.

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
                from :mod:`gemseo-calibration.scenario`.
            constraint_type: The type of constraint,
                `"eq"` for equality constraint and
                `"ineq"` for inequality constraint.
            constraint_name: The name of the constraint to be stored.
                If empty,
                the name of the constraint is generated from the output name.
            value: The value for which the constraint is active.
            positive: Whether to consider the inequality constraint as positive.
        """
        super().add_constraint(
            self.calibrator.add_measure(control_outputs)[0],
            constraint_type,
            constraint_name,
            value,
            positive,
        )

    @property
    def posts(self) -> list[str]:  # noqa: D102
        return (
            self.post_factory.class_names + self.__calibration_post_factory.class_names
        )

    def post_process(self, post_name: str, **options: Any) -> BasePost:  # noqa: D102
        if post_name in self.__calibration_post_factory.class_names:
            return self.__calibration_post_factory.execute(
                self.formulation.optimization_problem,
                self.calibrator.reference_data,
                self.prior_model_data,
                self.posterior_model_data,
                post_name,
                **options,
            )

        return super().post_process(post_name=post_name, **options)
