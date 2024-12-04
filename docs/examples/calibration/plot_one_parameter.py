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
"""# One calibration parameter.

This example illustrates the calibration of a discipline
with a poorly known parameter.
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
# Let us consider a function $f(x)=ax$
# from $\mathbb{R}$ to $\mathbb{R}$:
model = AnalyticDiscipline({"y": "a*x"}, name="model")

# %%
# This is a model of our reference data source,
# which is a kind of oracle providing input-output data
# without the mathematical relationship behind it:
reference = AnalyticDiscipline({"y": "2*x"}, name="reference")

# %%
# However in this pedagogical example,
# the mathematical relationship is known, and we can see that
# the parameter $a$ must be equal to 2
# so that the model and the reference are identical.
#
# In the following,
# we will try to find this value from a unique observation.

# %%
# Firstly,
# we have prior knowledge of the parameter values, that is $a\in[0,10]$:
prior = ParameterSpace()
prior.add_variable("a", lower_bound=0.0, upper_bound=10.0, value=0.0)

# %%
# Secondly,
# given an input space $[0,3]$:
input_space = DesignSpace()
input_space.add_variable("x", lower_bound=0.0, upper_bound=3.0)

# %%
# we generate reference output data by sampling the reference discipline:
reference_dataset = sample_disciplines(
    [reference], input_space, ["y"], algo_name="CustomDOE", samples=array([[1.0]])
)
reference_data = reference_dataset.to_dict_of_arrays(False)

# %%
# From this unique observation,
# we can build and execute a
# [CalibrationScenario][gemseo_calibration.scenario.CalibrationScenario]
# to find the value of the parameter $a$
# which minimizes a
# [BaseCalibrationMetric][gemseo_calibration.metrics.base_calibration_metric.BaseCalibrationMetric]
# taking into account the output $y$:
calibration = CalibrationScenario(
    model, "x", CalibrationMetricSettings(output_name="y", metric_name="MSE"), prior
)
calibration.execute(
    algo_name="NLOPT_COBYLA", reference_data=reference_data, max_iter=100
)

# %%
# Lastly,
# we can check that the calibrated parameter is very close to the expected one:
calibration.optimization_result.x_opt

# %%
# and plot an optimization history view:
calibration.post_process(post_name="OptHistoryView", save=False, show=True)

# %%
# as well as the model data versus the reference ones,
# before and after the calibration:
calibration.post_process(post_name="DataVersusModel", output="y", save=False, show=True)
