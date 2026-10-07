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
"""# Noisy observations.

This example illustrates the calibration of a discipline
with three poorly known parameters and from noisy observations.
"""

from __future__ import annotations

# %%
# Let us consider a model $f(x)=ax^2+bx+c$
# from $\mathbb{R}$ to $\mathbb{R}$:
from gemseo.discipline import AnalyticDiscipline
from gemseo.discipline import DisciplineChain
from gemseo.doe import OT_LHSC_Settings
from gemseo.doe import PYDOE_FULLFACT_Settings
from gemseo.formulation import DisciplinaryOpt_Settings
from gemseo.optimization import NLOPT_COBYLA_Settings
from gemseo.post import OptHistoryView_Settings
from gemseo.scenario import EvaluationScenario
from gemseo.scenario import EvaluationScenarioAdapter
from gemseo.space import DesignSpace
from gemseo.space import RandomSpace
from gemseo.uncertainty.distribution import OTNormalDistribution_Settings
from matplotlib import pyplot as plt
from numpy import array
from numpy import linspace

from gemseo_calibration.metrics.settings import CalibrationMetricSettings
from gemseo_calibration.scenario import CalibrationScenario

model = AnalyticDiscipline({"y": "a*x**2+b*x+c"}, name="model")

# %%
# This is a model of our reference data source,
# which is a kind of oracle providing input-output data
# without the mathematical relationship behind it:
original_model = AnalyticDiscipline({"y": "2*x**2-1.5*x+0.75"}, name="model")

reference = DisciplineChain([
    original_model,
    AnalyticDiscipline({"y": "y+u"}, name="noise"),
])
reference.set_cache(reference.CacheType.MEMORY_FULL)

# %%
# This reference model contains a random additive term $u$
# normally distributed with mean $\mu$ and standard deviation $\sigma$.
# This means that the observations of $f:x\mapsto 2x^2-1.5x+0.75$ are noised.

# %%
# In this pedagogical example,
# the mathematical relationship is known,
# and we can see that the parameters $a$, $b$ and $c$
# must be equal to 2, 0.5 and 0.75 respectively
# so that the model and the reference are identical.
#
# In the following,
# we will try to find these values from several information sources.

# %%
# Firstly,
# we have prior knowledge of the parameter values, that is $[a,b,c]\in[-5,5]^2$:
prior = DesignSpace()
prior.add_variable("a", lower_bound=-5.0, upper_bound=5.0, value=0.0)
prior.add_variable("b", lower_bound=-5.0, upper_bound=5.0, value=0.0)
prior.add_variable("c", lower_bound=-5.0, upper_bound=5.0, value=0.0)

# %%
# Secondly,
# we have reference output data over the input space $[0,3]$.
input_space = DesignSpace()
input_space.add_variable("x", lower_bound=0.0, upper_bound=3.0, value=1.5)
# %%
# These data are noisy; this noise can be modeled by a centered Gaussian random variable
# with standard deviation equal to 0.5.
noise_space = RandomSpace()
noise_space.add_variable("u", OTNormalDistribution_Settings(mu=0.0, sigma=0.5))

# %%
# The observations can be generated with two nested design of experiments:
# an inner one sampling the reference model $f$,
# an outer one repeating this sampling for different values of the noise.
# A classical way of doing this with |g| is to use an
# [EvaluationScenarioAdapter][gemseo.scenario.adapter.evaluation.EvaluationScenarioAdapter]
# which is a [Discipline][gemseo.core.discipline.discipline.Discipline] executing an
# [EvaluationScenario][gemseo.scenario.evaluation.EvaluationScenario]
# for a given value of $u$.
# For example,
# let us imagine an
# [EvaluationScenario][gemseo.scenario.evaluation.EvaluationScenario]
# evaluating the reference data source at 5 equispaced points $x_1,\ldots,x_5$.

sub_scenario = EvaluationScenario(
    [reference], input_space, formulation_settings=DisciplinaryOpt_Settings()
)
sub_scenario.add_observable("y")
sub_scenario.set_algorithm(PYDOE_FULLFACT_Settings(n_samples=5))

adapter = EvaluationScenarioAdapter(sub_scenario, ["u"], ["y"])

# %%
# Then,
# this
# [EvaluationScenarioAdapter][gemseo.scenario.adapter.evaluation.EvaluationScenarioAdapter]
# is embedded in an
# [EvaluationScenario][gemseo.scenario.evaluation.EvaluationScenario]
# in charge to sample it over the uncertain space.
scenario = EvaluationScenario(
    [adapter], noise_space, formulation_settings=DisciplinaryOpt_Settings()
)
scenario.add_observable("y")
scenario.execute(OT_LHSC_Settings(n_samples=5))
reference_data = reference.cache.to_dataset().to_dict_of_arrays(False)

# %%
# From these information sources,
# we can build and execute a
# [CalibrationScenario][gemseo_calibration.scenario.CalibrationScenario]
# to find the values of the parameters $a$, $b$ and $c$
# which minimize a
# [BaseCalibrationMetric][gemseo_calibration.metrics.base_calibration_metric.BaseCalibrationMetric]
# related to the output $y$:
calibration = CalibrationScenario(
    model, "x", CalibrationMetricSettings(output_name="y", metric_name="MSE"), prior
)
calibration.execute(NLOPT_COBYLA_Settings(max_iter=100), reference_data=reference_data)

# %%
# Lastly,
# we can see that the calibrated parameters are different from the expected ones
calibration.optimization_result.x_opt

# %%
# even if the result are converged:
calibration.post_process(OptHistoryView_Settings(save=False, show=True))

# %%
# However,
# the calibrated model is close the expected one:
expression = "a*x**2+b*x+c"
for parameter_name, parameter_value in calibration.posterior_parameters.items():
    expression = expression.replace(parameter_name, str(parameter_value[0]))
calibrated = AnalyticDiscipline({"y": expression}, name="calibrated")

x_values = linspace(0.0, 3.0, 100)
y_values = [original_model.execute({"x": array([x_i])})["y"][0] for x_i in x_values]
post_y_values = [calibrated.execute({"x": array([x_i])})["y"][0] for x_i in x_values]
plt.plot(x_values, y_values, color="blue", label="Unknown model")
plt.plot(x_values, post_y_values, color="red", label="Calibrated model")

plt.plot(
    reference_data["x"][:, 0],
    reference_data["y"][:, 0],
    color="blue",
    linestyle="",
    marker="x",
    label="Reference data",
)
plt.legend()
plt.show()
