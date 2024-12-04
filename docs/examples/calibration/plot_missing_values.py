# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This work is licensed under a BSD 0-Clause License.
#
# Permission to use, copy, modify, and/or distribute this software
# for any purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL
# WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
# THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT,
# OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING
# FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
# WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
"""# Observations with missing values.

This example illustrates the calibration of a discipline
with two poorly known parameters and from observations with missing values.
"""

from __future__ import annotations

from gemseo.algos.parameter_space import ParameterSpace
from gemseo.datasets.dataset import Dataset
from gemseo.disciplines.analytic import AnalyticDiscipline
from numpy import array
from numpy import nan

from gemseo_calibration.metrics.settings import CalibrationMetricSettings
from gemseo_calibration.scenario import CalibrationScenario

model = AnalyticDiscipline({"y": "a*x", "z": "b*x"}, name="model")

prior = ParameterSpace()
prior.add_variable("a", lower_bound=0.0, upper_bound=10.0, value=0.0)
prior.add_variable("b", lower_bound=0.0, upper_bound=10.0, value=0.0)

data = array([
    [1, 1.0, 2.0, nan],
    [2, 1.0, nan, 3.0],
    [3, 2.0, 4.0, nan],
    [4, 2.0, nan, 6.0],
])
reference_data = Dataset.from_array(
    data,
    variable_names=["index", "x", "y", "z"],
    variable_names_to_group_names={
        "index": "inputs",
        "x": "inputs",
        "y": "outputs",
        "z": "outputs",
    },
).to_dict_of_arrays(False)

metric_settings = [
    CalibrationMetricSettings(output_name="y", metric_name="MSE"),
    CalibrationMetricSettings(output_name="z", metric_name="MSE"),
]
calibration = CalibrationScenario(model, "x", metric_settings, prior)
calibration.execute(
    algo_name="NLOPT_COBYLA", reference_data=reference_data, max_iter=100
)
