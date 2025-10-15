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
"""# Functional output discretized on a mesh.

This example illustrates the calibration of a discipline
with two poorly known parameters and a functional output discretized on a mesh.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo import sample_disciplines
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.discipline.discipline import Discipline
from numpy import array
from numpy import linspace

from gemseo_calibration.metrics.settings import CalibrationMetricSettings
from gemseo_calibration.scenario import CalibrationScenario

if TYPE_CHECKING:
    from gemseo.typing import StrKeyMapping

# %%
# Let us consider a model $f(x)=[ax,\gamma bx, \gamma]$
# from $\mathbb{R}$ to $\mathbb{R}^{11}$
# where $\gamma=[0,0.25,0.5,0.75,1.]$ plays the role of a mesh.
# In practice,
# we could imagine a model having an output related to a mesh $\gamma$
# whose size and nodes would depend on the model inputs.
# Thus, this mesh is also an output of the model.


class Model(Discipline):
    def __init__(self) -> None:
        super().__init__()
        self.input_grammar.update_from_names(["x", "a", "b"])
        self.output_grammar.update_from_names(["y", "z", "mesh"])
        self.default_input_data = {
            "x": array([0.0]),
            "a": array([0.0]),
            "b": array([0.0]),
        }

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        x_input = input_data["x"]
        a_parameter = input_data["a"]
        b_parameter = input_data["b"]
        y_output = a_parameter * x_input
        z_mesh = linspace(0, 1, 5)
        z_output = b_parameter * x_input[0] * z_mesh
        return {"y": y_output, "z": z_output, "mesh": z_mesh}


# %%
# This is a model of our reference data source,
# which is a kind of oracle providing input-output data
# without the mathematical relationship behind it:
class ReferenceModel(Discipline):
    def __init__(self) -> None:
        super().__init__()
        self.input_grammar.update_from_names(["x"])
        self.output_grammar.update_from_names(["y", "z", "mesh"])
        self.default_input_data = {"x": array([0.0])}

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        x_input = input_data["x"]
        y_output = 2 * x_input
        z_mesh = linspace(0, 1, 5)
        z_output = 3 * x_input[0] * z_mesh
        return {"y": y_output, "z": z_output, "mesh": z_mesh}


# %%
# However in this pedagogical example,
# the mathematical relationship is known, and we can see that
# the parameters $a$ and $b$ must be equal to 2 and 3 respectively
# so that the model and the reference are identical.
#
# In the following,
# we will try to find these values from several information sources.

# %%
# Firstly,
# we have prior knowledge of the parameter values, that is $[a,b]\in[0,10]^2$:
prior = ParameterSpace()
prior.add_variable("a", lower_bound=0.0, upper_bound=10.0, value=0.0)
prior.add_variable("b", lower_bound=0.0, upper_bound=10.0, value=0.0)

# %%
# Secondly,
# given an input space $[0,3]$:
input_space = DesignSpace()
input_space.add_variable("x", lower_bound=0.0, upper_bound=3.0)

# %%
# we generate reference output data by sampling the reference discipline:
reference = ReferenceModel()
reference_dataset = sample_disciplines(
    [reference],
    input_space,
    ["mesh", "y", "z"],
    algo_name="CustomDOE",
    samples=array([[1.0], [2.0]]),
)
reference_data = reference_dataset.to_dict_of_arrays(False)

# %%
# From these information sources,
# we can build and execute a
# [CalibrationScenario][gemseo_calibration.scenario.CalibrationScenario]
# to find the values of the parameters $a$ and $b$
# which minimize a
# [BaseCalibrationMetric][gemseo_calibration.metrics.base_calibration_metric.BaseCalibrationMetric]
# taking into account the outputs $y$ and $z$:
model = Model()
metric_settings = [
    CalibrationMetricSettings(output_name="y", metric_name="MSE"),
    CalibrationMetricSettings(output_name="z", metric_name="ISE", mesh_name="mesh"),
]
calibration = CalibrationScenario(model, "x", metric_settings, prior)
calibration.execute(
    algo_name="NLOPT_COBYLA", reference_data=reference_data, max_iter=100
)

# %%
# Lastly,
# we can check that the calibrated parameters are very close to the expected ones:
calibration.optimization_result.x_opt

# %%
# and plot an optimization history view:
calibration.post_process(post_name="OptHistoryView", save=False, show=True)

# %%
# as well as the model data versus the reference ones,
# before and after the calibration:
calibration.post_process(post_name="DataVersusModel", output="y", save=False, show=True)
