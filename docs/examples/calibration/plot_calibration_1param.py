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
"""# Calibration scenario."""

from __future__ import annotations

from gemseo.algos.parameter_space import ParameterSpace
from gemseo.disciplines.analytic import AnalyticDiscipline
from numpy import array

from gemseo_calibration.scenario import CalibrationMeasure
from gemseo_calibration.scenario import CalibrationScenario

# %%
# Let us consider a function $f(x)=ax$
# from $\mathbb{R}$ to $\mathbb{R}$:
model = AnalyticDiscipline({"y": "a*x"}, name="model")

# %%
# This is a model of our reference data source,
# which a kind of oracle providing input-output data
# without the mathematical relationship behind it:
reference = AnalyticDiscipline({"y": "2*x"}, name="reference")

# %%
# However in this pedagogical example,
# the mathematical relationship is known and we can see that
# the parameter $a$ must be equal to 2
# so that the model and the reference are identical.
#
# In the following,
# we will try to find this value from an unique observation.

# %%
# Firstly,
# we have a prior information about the parameters, that is $a\in[0,10]$:
prior = ParameterSpace()
prior.add_variable("a", lower_bound=0.0, upper_bound=10.0, value=0.0)

# %%
# Secondly,
# we have reference output data over the input space $[0.,3.]$:
reference.set_cache(reference.CacheType.MEMORY_FULL)
reference.execute({"x": array([1.0])})
reference_data = reference.cache.to_dataset().to_dict_of_arrays(False)

# %%
# From this unique observation,
# we can build and execute a
# [CalibrationScenario][gemseo_calibration.scenario.CalibrationScenario]
# to find the value of the parameter $a$
# which minimizes a
# [CalibrationMeasure][gemseo_calibration.measure.CalibrationMeasure]
# taking into account the outputs $y$:
calibration = CalibrationScenario(model, "x", CalibrationMeasure("y", "MSE"), prior)
calibration.execute({
    "algo": "NLOPT_COBYLA",
    "reference_data": reference_data,
    "max_iter": 100,
})

# %%
# Lastly,
# we get the calibrated parameters:

# %%
# plot an optimization history view:
calibration.post_process("OptHistoryView", save=False, show=True)

# %%
# as well as the model data versus the reference ones,
# before and after the calibration:
calibration.post_process("DataVersusModel", output="y", save=False, show=True)
