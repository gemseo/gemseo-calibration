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
"""
Calibration scenario with a mesh-based output
=============================================
"""
from __future__ import annotations

from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.discipline import MDODiscipline
from gemseo_calibration.scenario import CalibrationMeasure
from gemseo_calibration.scenario import CalibrationScenario
from numpy import array
from numpy import linspace

#######################################################################################
# Let us consider a function :math:`f(x)=[ax,\gamma bx, \gamma]`
# from :math:`\mathbb{R}` to :math:`\mathbb{R}^11`
# where :math:`\gamma=[0,0.25,0.5,0.75,1.]` plays the role of a mesh.
# In practice,
# we could imagine a model having an output related to a mesh :math:`\gamma`
# whose size and nodes would depend on the model inputs.
# Thus, this mesh is also an output of the model.


class Model(MDODiscipline):
    def __init__(self):
        super().__init__()
        self.input_grammar.initialize_from_data_names(["x", "a", "b"])
        self.output_grammar.initialize_from_data_names(["y", "z", "mesh"])
        self.default_inputs = {"x": array([0.0]), "a": array([0.0]), "b": array([0.0])}

    def _run(self):
        x_input = self.local_data["x"]
        a_parameter = self.local_data["a"]
        b_parameter = self.local_data["b"]
        y_output = a_parameter * x_input
        z_mesh = linspace(0, 1, 5)
        z_output = b_parameter * x_input[0] * z_mesh
        self.store_local_data(y=y_output, z=z_output, mesh=z_mesh)


#######################################################################################
# This is a model of a our reference data source,
# which a kind of oracle providing input-output data
# without the mathematical relationship behind it:
class ReferenceModel(MDODiscipline):
    def __init__(self):
        super().__init__()
        self.input_grammar.initialize_from_data_names(["x"])
        self.output_grammar.initialize_from_data_names(["y", "z", "mesh"])
        self.default_inputs = {"x": array([0.0])}

    def _run(self):
        x_input = self.local_data["x"]
        y_output = 2 * x_input
        z_mesh = linspace(0, 1, 5)
        z_output = 3 * x_input[0] * z_mesh
        self.store_local_data(y=y_output, z=z_output, mesh=z_mesh)


#######################################################################################
# However in this pedagogical example,
# the mathematical relationship is known and we can see that
# the parameters :math:`a` and :math:`b` must be equal to 2 and 3 respectively
# so that the model and the reference are identical.
#
# In the following,
# we will try to find these values from several information sources.

#######################################################################################
# Firstly,
# we have a prior information about the parameters, that is :math:`[a,b]\in[0,10]^2`:
prior = ParameterSpace()
prior.add_variable("a", l_b=0.0, u_b=10.0, value=0.0)
prior.add_variable("b", l_b=0.0, u_b=10.0, value=0.0)

#######################################################################################
# Secondly,
# we have reference output data over the input space :math:`[0.,3.]`:
reference = ReferenceModel()
reference.set_cache_policy("MemoryFullCache")
reference.execute({"x": array([1.0])})
reference.execute({"x": array([2.0])})
reference_data = reference.cache.export_to_dataset(by_group=True)

#######################################################################################
# From these information sources,
# we can build and execute a :class:`.CalibrationScenario`
# to find the values of the parameters :math:`a` and :math:`b`
# which minimizes a :class:`.CalibrationMeasure`
# taking into account the outputs :math:`y` and :math:`z`:
model = Model()
control_outputs = [
    CalibrationMeasure("y", "MSE"),
    CalibrationMeasure("z", "ISE", "mesh"),
]
calibration = CalibrationScenario(model, "x", control_outputs, prior)
calibration.execute(
    {"algo": "NLOPT_COBYLA", "reference_data": reference_data, "max_iter": 100}
)

#######################################################################################
# Lastly,
# we get the calibrated parameters:
print("Initial parameters: ", calibration.prior_parameters)
print("Calibrated parameters: ", calibration.posterior_parameters)

#######################################################################################
# plot an optimization history view:
calibration.post_process("OptHistoryView", save=False, show=True)

#######################################################################################
# as well as the model data versus the reference ones,
# before and after the calibration:
calibration.post_process("DataVersusModel", output="y", save=False, show=True)
