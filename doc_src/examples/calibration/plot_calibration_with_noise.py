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
"""Calibration scenario with noised observations.
=============================================
"""

#######################################################################################
# Let us consider a function :math:`f(x)=ax^2+bx+c`
# from :math:`\mathbb{R}` to :math:`\mathbb{R}`:
from __future__ import annotations

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.chain import MDOChain
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.disciplines.scenario_adapters.mdo_scenario_adapter import MDOScenarioAdapter
from gemseo.scenarios.doe_scenario import DOEScenario
from matplotlib import pyplot as plt
from numpy import array
from numpy import linspace

from gemseo_calibration.scenario import CalibrationMeasure
from gemseo_calibration.scenario import CalibrationScenario

model = AnalyticDiscipline({"y": "a*x**2+b*x+c"}, name="model")

#######################################################################################
# This is a model of a our reference data source,
# which a kind of oracle providing input-output data
# without the mathematical relationship behind it:
original_model = AnalyticDiscipline({"y": "2*x**2-1.5*x+0.75"}, name="model")

reference = MDOChain([original_model, AnalyticDiscipline({"y": "y+u"}, name="noise")])
reference.set_cache_policy(reference.CacheType.MEMORY_FULL)

#######################################################################################
# This reference model contains a random additive term :math:`u`
# normally distributed with mean :math:`\mu` and standard deviation :math:`\sigma`.
# This means that the observations of :math:`f:x\mapsto 2*x^2-0.5*x` are noised.

#######################################################################################
# In this pedagogical example,
# the mathematical relationship is known
# and we can see that the parameters :math:`a`, :math:`b` and :math:`c`
# must be equal to 2, 0.5 and 0.75 respectively
# so that the model and the reference are identical.
#
# In the following,
# we will try to find these values from several information sources.

#######################################################################################
# Firstly,
# we have a prior information about the parameters, that is :math:`[a,b,c]\in[-5,5]^2`:
prior = ParameterSpace()
prior.add_variable("a", l_b=-5.0, u_b=5.0, value=0.0)
prior.add_variable("b", l_b=-5.0, u_b=5.0, value=0.0)
prior.add_variable("c", l_b=-5.0, u_b=5.0, value=0.0)

#######################################################################################
# Secondly,
# we have reference output data over the input space :math:`[0.,3.]`.
input_space = DesignSpace()
input_space.add_variable("x", l_b=0.0, u_b=3.0, value=1.5)
#######################################################################################
# These data are noisy; this noise can be modeled by a centered Gaussian random variable
# with standard deviation equal to 0.5.
noise_space = ParameterSpace()
noise_space.add_random_variable("u", "OTNormalDistribution", mu=0.0, sigma=0.5)

#######################################################################################
# The observations can be generated with two nested design of experiments:
# an inner one sampling the reference model :math:`f`,
# an outer one repeating this sampling for different values of the noise.
# A classical way of doing this with |g| is to use a :class:`.MDOScenarioAdapter`,
# which is a :class:`.MDODiscipline` executing a :class:`.DOEScenario`
# for a given value of :math:`u`.
# For example, let us imagine a :class:`.DOEScenario`
# evaluating the reference data source at 5 equispaced points :math:`x_1,\ldots,x_5`.

sub_scenario = DOEScenario([reference], "DisciplinaryOpt", "y", input_space)
sub_scenario.default_inputs = {"algo": "fullfact", "n_samples": 5}

adapter = MDOScenarioAdapter(sub_scenario, ["u"], ["y"])

#######################################################################################
# Then,
# this :class:`.MDOScenarioAdapter` is embedded in a :class:`.DOEScenario`
# in charge to sample it over the uncertain space.
scenario = DOEScenario([adapter], "DisciplinaryOpt", "y", noise_space)
scenario.execute({"algo": "OT_LHSC", "n_samples": 5})
reference_data = reference.cache.to_dataset().to_dict_of_arrays(False)

#######################################################################################
# From these information sources,
# we can build and execute a :class:`.CalibrationScenario`
# to find the value of the parameters :math:`a`, :math:`b` and :math:`c`
# which minimizes a :class:`.CalibrationMeasure` related to the output :math:`y`:
calibration = CalibrationScenario(model, "x", CalibrationMeasure("y", "MSE"), prior)
calibration.execute({
    "algo": "NLOPT_COBYLA",
    "reference_data": reference_data,
    "max_iter": 100,
})

#######################################################################################
# Lastly,
# we get the calibrated parameters:

#######################################################################################
# plot an optimization history view:
calibration.post_process("OptHistoryView", save=False, show=True)

#######################################################################################
# as well as the model data versus the reference ones:
expression = "a*x**2+b*x+c"
for parameter_name, parameter_value in calibration.posterior_parameters.items():
    expression = expression.replace(parameter_name, str(parameter_value[0]))
calibrated = AnalyticDiscipline({"y": expression}, name="calibrated")

x_values = linspace(0.0, 3.0, 100)
y_values = [original_model.execute({"x": array([x_i])})["y"][0] for x_i in x_values]
post_y_values = [calibrated.execute({"x": array([x_i])})["y"][0] for x_i in x_values]
plt.plot(x_values, y_values, color="blue", label="Unknown model")
plt.plot(x_values, post_y_values, color="red", label="Calibrated model")

x_points = []
y_points = []
for data in reference.cache:
    x_points.append(data.inputs["x"][0])
    y_points.append(data.outputs["y"][0])

plt.plot(
    x_points,
    y_points,
    color="blue",
    linestyle="",
    marker="x",
    label="Reference data",
)
plt.legend()
plt.show()
