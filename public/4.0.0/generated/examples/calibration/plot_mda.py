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
"""# Strongly coupled models.

This example illustrates the calibration of a system of strongly coupled disciplines
with two poorly known parameters.
"""

from __future__ import annotations

from gemseo import sample_disciplines
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from numpy import array

from gemseo_calibration.metrics.settings import CalibrationMetricSettings
from gemseo_calibration.scenario import CalibrationScenario

# %%
# Let us consider a model $f(x)$ from $\mathbb{R}$ to $\mathbb{R}^2$,
# decomposed into a model $f_1(x,y_2)=x+ay_2$ from $\mathbb{R}^2$ to $\mathbb{R}$:
model_1 = AnalyticDiscipline({"y1": "x+a*y2"}, name="model_1")

# %%
# and a model $f_2(x,y_1)=x+by_1$ from $\mathbb{R}^2$ to $\mathbb{R}$:
model_2 = AnalyticDiscipline({"y2": "x+b*y1"}, name="model_2")
models = [model_1, model_2]

# %%
# such that $f(x)=(y_1,y_2).$
#
# This is a model of our reference data source,
# which is a kind of oracle providing input-output data
# without the mathematical relationship behind it:
reference = AnalyticDiscipline({"y1": "x+2*y2", "y2": "x+3*y1"}, name="reference")

# %%
# However in this pedagogical example,
# the mathematical relationship is known, and we can see that
# the parameters $a$ and $b$ must be equal to 2 and 3 respectively
# so that the model and the reference are identical.
#
# In the following,
# we will try to find this value from a unique observation.

# %%
# Firstly,
# we have prior knowledge of the parameter values, that is $[a,b]\in[1,10]^2$:
prior = ParameterSpace()
prior.add_variable("a", lower_bound=1.0, upper_bound=10.0, value=1.0)
prior.add_variable("b", lower_bound=1.0, upper_bound=10.0, value=1.0)

# %%
# Secondly,
# given an input space $[0,3]$:
input_space = DesignSpace()
input_space.add_variable("x", lower_bound=0.0, upper_bound=3.0)

# %%
# we generate reference output data by sampling the reference discipline:
reference_dataset = sample_disciplines(
    [reference],
    input_space,
    ["y1", "y2"],
    algo_name="CustomDOE",
    samples=array([[1.0], [2.0]]),
)
reference_data = reference_dataset.to_dict_of_arrays(False)

# %%
# From this unique observation,
# we can build and execute a
# [CalibrationScenario][gemseo_calibration.scenario.CalibrationScenario]
# to find the value of the parameters $a$ and $b$
# which minimizes
# [CalibrationMetric][gemseo_calibration.metrics.base_calibration_metric.BaseCalibrationMetric]
# taking into account the outputs $y_1$ and $y_2$:
calibration = CalibrationScenario(
    models,
    "x",
    [
        CalibrationMetricSettings(output_name="y1", metric_name="MSE"),
        CalibrationMetricSettings(output_name="y2", metric_name="MSE"),
    ],
    prior,
)
calibration.execute(
    algo_name="NLOPT_COBYLA", reference_data=reference_data, max_iter=100
)

# %%
# Lastly,
# we can check that the calibrated parameters are very close to the expected ones
calibration.optimization_result.x_opt

# %%
# and plot an optimization history view:
calibration.post_process(post_name="OptHistoryView", save=False, show=True)

# %%
# as well as the model data versus the reference ones,
# before and after the calibration:
calibration.post_process(
    post_name="DataVersusModel", output="y1", save=False, show=True
)
