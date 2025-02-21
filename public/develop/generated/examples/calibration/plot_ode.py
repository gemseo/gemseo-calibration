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
"""Calibrate an ordinary differential equation."""

from gemseo.algos.design_space import DesignSpace
from matplotlib import pyplot as plt
from numpy import atleast_2d
from numpy import full
from numpy import linspace

from gemseo_calibration.metrics.settings import CalibrationMetricSettings
from gemseo_calibration.problems.signal.oscillator import Oscillator
from gemseo_calibration.scenario import CalibrationScenario
from gemseo_calibration.signal.signal_generator_discipline import (
    SignalGeneratorDiscipline,
)

# %%
# ## The problem
#
# ### The data generator
#
# In this example,
# we will observe a time-dependent angular rate oscillator,
# or more precisely a reference model generating synthetic data.
# Its state $s=(p,v)$,
# defined by its position $p$ and its velocity $v$,
# is modelled by the ordinary differential equation (ODE)
#
# $$\frac{ds(t)}{dt}=(v(t),-\omega^2(t) p(t))$$
#
# with $p(t_0)=p_0$, $v(t_0)=v_0$ and $\omega(t_0)=\omega_0$ at initial time $t_0$:
# Its angular velocity $\omega$ is also modelled by an ODE:
#
# $$\frac{d\omega(t)}{dt} = -a\exp(-at)$$
#
# with $a=0.02$ and $\omega(t_0)=\omega_0$.
#
# This oscillator can be built as follows:
omega_0 = 1.0
p_0 = 1.0
v_0 = 0.0
oscillator = Oscillator()
# %%
# We then seek to simulate the time evolution of its state from observations,
# under the assumption that the model of $\omega$ is unknown.
#
# ### The simulator
#
# We can approximate this oscillator by a simulator based on an oscillator model
# whose angular velocity is constant and set to $\omega_0$:
simulator = Oscillator(omega=omega_0)
# %%
# Unfortunately,
# this approximation is too crude, as illustrated
# over the time interval $[t_0,t_n]=[0,20]$:
t_0 = 0.0
t_n = 20.0
n_nodes = 200
times = linspace(t_0, t_n, num=n_nodes)
data = oscillator.generate(times, {"omega": omega_0, "position": p_0, "velocity": v_0})
simulations = simulator.generate(times, {"position": p_0, "velocity": v_0})

plt.plot(
    data.times, data.evolution["position"], "-", color="tab:blue", label="Oscillator"
)
plt.plot(
    simulations.times,
    simulations.evolution["position"],
    "--",
    color="tab:orange",
    label="Simulator",
)
plt.grid()
plt.legend()
plt.ylabel("Position (m)")
_ = plt.xlabel("Time (s)")
# %%
# This difference is perfectly understandable,
# since if we look at the observations of the oscillator,
# we can deduce that its angular velocity evolves with time.
# It is therefore in our interest to take this information into account,
# either by changing its value regularly or by using a parametric model for $\omega$.
#
# ## Periodic calibration
#
# ### The methodology
# We therefore seek to calibrate this simulator:
calibrated_simulator = Oscillator(omega=omega_0)
# %%
# at $m=10$ times $(t_i)_{1\leq i \leq m}$ equally spaced with time step $\delta$
# over the interval $[t_0,t_n]$:
m = 10
n = m + 1
t_i = t_0
delta = (t_n - t_0) / n
n_sub_nodes = n_nodes // n
# %%
# by varying $\omega$ over the interval $[0.5,3]$:
search_space = DesignSpace()
search_space.add_variable("omega", lower_bound=0.5, upper_bound=3.0, value=omega_0)
# %%
# in order to minimize
# the mean squared error of the position of the `calibrated_simulator`.
output_name = "position"
# %%
# In the following,
# we mimic the sequential acquisition of observations from the `oscillator`
# using a for-loop in which we use a
# [CalibrationScenario][gemseo_calibration.scenario.CalibrationScenario]
# to calibrate the `calibrated_simulator` every $\delta$ seconds
# and execute the `simulator` with the constant angular velocity.
# First,
# we initialize the state variables of the ODEs:
initial_oscillator_omega = omega_0
initial_oscillator_position = p_0
initial_oscillator_velocity = v_0
initial_simulator_position = p_0
initial_simulator_velocity = v_0
initial_calibrated_simulator_position = p_0
initial_calibrated_simulator_velocity = v_0
new_omega = [full(n_sub_nodes, omega_0), None]
# %%
# and then,
# at each calibration time $t_i$,
# we
#
# 1. plot the observations acquired by `oscillator` over $[t_{i-1},t_i]$,
# 2. generate simulations with the `simulator`  over $[t_{i-1},t_i]$ and plot them,
# 3. calibrate the `calibrated_simulator` over $[t_{i-1},t_i]$,
#    generate simulations and plot them.
fig, (ax1, ax2) = plt.subplots(2, 1)
for i in range(m + 1):
    # The times of interest, from the initial time to the calibration time.
    times = linspace(t_i, t_i + delta, num=n_sub_nodes)

    # Generation of reference data at times of interest.
    oscillator_signal = oscillator.generate(
        times,
        {
            "omega": initial_oscillator_omega,
            "position": initial_oscillator_position,
            "velocity": initial_oscillator_velocity,
        },
    )

    # Generation of simulations at times of interest with the uncalibrated simulator.
    simulator_signal = simulator.generate(
        times,
        {
            "position": initial_simulator_position,
            "velocity": initial_simulator_velocity,
        },
    )
    # Calibration of the simulator at times of interest.
    calibration = CalibrationScenario(
        SignalGeneratorDiscipline(
            # The signal generator
            calibrated_simulator,
            # The state variable names
            ["position", "velocity"],
            # The parameter names
            ["omega"],
            # The observable names
            [output_name],
            # The times of interest
            times,
        ),
        # The input variable names in the calibration data
        ["initial_position", "initial_velocity"],
        CalibrationMetricSettings(output_name=output_name, metric_name="MSE"),
        # The possible values of omega
        search_space,
    )
    calibration.execute(
        algo_name="NLOPT_COBYLA",
        reference_data={
            "initial_position": atleast_2d(initial_oscillator_position),
            "initial_velocity": atleast_2d(initial_oscillator_velocity),
            output_name: atleast_2d(oscillator_signal.evolution[output_name]),
        },
        max_iter=100,
    )

    # Generation of simulations at times of interest with the calibrated simulator.
    omega_opt = calibration.optimization_result.x_opt_as_dict["omega"][0]
    calibrated_simulator.rhs_discipline.default_input_data["omega"] = omega_opt
    calibrated_simulator_signal = calibrated_simulator.generate(
        times,
        {
            "position": initial_calibrated_simulator_position,
            "velocity": initial_calibrated_simulator_velocity,
        },
    )

    new_omega = (full(n_sub_nodes, omega_opt), new_omega[0])

    # Plot the results at times of interest.
    ax1.plot(
        times,
        oscillator_signal.evolution[output_name],
        "-",
        color="tab:blue",
        label="Oscillator" if i == 0 else "",
    )
    ax1.plot(
        times,
        simulator_signal.evolution[output_name],
        "--",
        color="tab:orange",
        label="Simulator" if i == 0 else "",
    )
    ax1.plot(
        times,
        calibrated_simulator_signal.evolution[output_name],
        "-.",
        color="tab:green",
        label="Calibrated simulator" if i == 0 else "",
    )
    ax1.axvline(x=times[-1], color="gray", label="Calibration" if i == 0 else "")
    ax2.plot(
        times,
        oscillator_signal.evolution["omega"],
        "-",
        color="tab:blue",
    )
    ax2.plot(
        times,
        full(n_sub_nodes, omega_0),
        "--",
        color="tab:orange",
    )
    ax2.plot(
        times,
        new_omega[1],
        "-.",
        color="tab:green",
    )
    ax2.axvline(x=times[-1], color="gray")

    # Update the initial values of the state variables for the next iteration.
    t_i += delta
    # --- For the reference data.
    final_state = oscillator_signal.final
    initial_oscillator_omega = final_state["omega"][0]
    initial_oscillator_position = final_state["position"][0]
    initial_oscillator_velocity = final_state["velocity"][0]
    # --- For the uncalibrated simulator.
    final_state = simulator_signal.final
    initial_simulator_position = final_state["position"][0]
    initial_simulator_velocity = final_state["velocity"][0]
    # --- For the calibrated simulator.
    final_state = calibrated_simulator_signal.final
    initial_calibrated_simulator_position = final_state["position"][0]
    initial_calibrated_simulator_velocity = final_state["velocity"][0]

# Finalize the plot.
ax1.legend()
ax1.set_ylabel("Position (m)")
ax2.set_ylabel(r"Angular velocity ($s^{-1}$)")
ax2.set_xlabel("Time (s)")
ax1.grid()
ax2.grid()

# %%
# ### The results
# In this graph,
# the vertical lines indicate the calibration times.
# We can see that
# the original simulator diverges from observations as time progresses,
# whereas the recalibrated simulator is close to observations
# throughout the time horizon.
# From an $\omega$ point of view,
# we can see that the stairs function obtained with the recalibrated simulator
# is close to the expected one, unlike that of the original model.
#
# ## One-shot calibration
#
# ### The methodology
#
# We could also imagine the case where
# the ODE defining the angular velocity is a grey box,
# in the sense that only the parameter `a` is unknown.
# The above then becomes:
calibrated_simulator = Oscillator()

search_space = DesignSpace()
search_space.add_variable("a", lower_bound=0.0001, upper_bound=0.1, value=0.001)

initial_oscillator_omega = omega_0
initial_oscillator_position = p_0
initial_oscillator_velocity = v_0
initial_simulator_position = p_0
initial_simulator_velocity = v_0
initial_calibrated_simulator_omega = omega_0
initial_calibrated_simulator_position = p_0
initial_calibrated_simulator_velocity = v_0
fig, (ax1, ax2) = plt.subplots(2, 1)
for i in range(m + 1):
    # The times of interest, from the initial time to the calibration time.
    times = linspace(t_i, t_i + delta, num=n_sub_nodes)

    # Generation of reference data at times of interest.
    oscillator_signal = oscillator.generate(
        times,
        {
            "omega": initial_oscillator_omega,
            "position": initial_oscillator_position,
            "velocity": initial_oscillator_velocity,
        },
    )

    # Generation of simulations at times of interest with the uncalibrated simulator.
    simulator_signal = simulator.generate(
        times,
        {
            "position": initial_simulator_position,
            "velocity": initial_simulator_velocity,
        },
    )

    # Calibration of the simulator at times of interest.
    calibration = CalibrationScenario(
        SignalGeneratorDiscipline(
            # The signal generator
            calibrated_simulator,
            # The state variable names
            ["omega", "position", "velocity"],
            # The parameter names
            ["a"],
            # The observable names
            [output_name],
            # The times of interest
            times,
        ),
        # The input variable names in the calibration data
        ["initial_omega", "initial_position", "initial_velocity"],
        CalibrationMetricSettings(output_name=output_name, metric_name="MSE"),
        # The possible values of a
        search_space,
    )
    calibration.execute(
        algo_name="NLOPT_COBYLA",
        reference_data={
            "initial_omega": atleast_2d(initial_oscillator_omega),
            "initial_position": atleast_2d(initial_oscillator_position),
            "initial_velocity": atleast_2d(initial_oscillator_velocity),
            output_name: atleast_2d(oscillator_signal.evolution[output_name]),
        },
        max_iter=100,
    )

    # Generation of simulations at times of interest with the calibrated simulator.
    a_opt = calibration.optimization_result.x_opt_as_dict["a"][0]
    calibrated_simulator.rhs_discipline.default_input_data["a"] = a_opt
    calibrated_simulator_signal = calibrated_simulator.generate(
        times,
        {
            "omega": initial_calibrated_simulator_omega,
            "position": initial_calibrated_simulator_position,
            "velocity": initial_calibrated_simulator_velocity,
        },
    )

    # Plot the results at times of interest.
    ax1.plot(
        times,
        oscillator_signal.evolution[output_name],
        "-",
        color="tab:blue",
        label="Oscillator" if i == 0 else "",
    )
    ax1.plot(
        times,
        simulator_signal.evolution[output_name],
        "--",
        color="tab:orange",
        label="Simulator" if i == 0 else "",
    )
    ax1.plot(
        times,
        calibrated_simulator_signal.evolution[output_name],
        "-.",
        color="tab:green",
        label="Calibrated simulator" if i == 0 else "",
    )
    ax1.axvline(x=times[-1], color="gray", label="Calibration" if i == 0 else "")
    ax2.plot(
        times,
        oscillator_signal.evolution["omega"],
        "-",
        color="tab:blue",
    )
    ax2.plot(
        times,
        full(n_sub_nodes, omega_0),
        "--",
        color="tab:orange",
    )
    ax2.plot(
        times,
        calibrated_simulator_signal.evolution["omega"],
        "-.",
        color="tab:green",
    )
    ax2.axvline(x=times[-1], color="gray")

    # Update the initial values of the state variables for the next iteration.
    t_i += delta
    # --- For the reference data.
    final_state = oscillator_signal.final
    initial_oscillator_omega = final_state["omega"][0]
    initial_oscillator_position = final_state["position"][0]
    initial_oscillator_velocity = final_state["velocity"][0]
    # --- For the uncalibrated simulator.
    final_state = simulator_signal.final
    initial_simulator_position = final_state["position"][0]
    initial_simulator_velocity = final_state["velocity"][0]
    # --- For the calibrated simulator.
    final_state = calibrated_simulator_signal.final
    initial_calibrated_simulator_omega = final_state["omega"][0]
    initial_calibrated_simulator_position = final_state["position"][0]
    initial_calibrated_simulator_velocity = final_state["velocity"][0]

# Finalize the plot
ax1.legend()
ax1.set_ylabel("Position (m)")
ax2.set_ylabel(r"Angular velocity ($s^{-1}$)")
ax2.set_xlabel("Time (s)")
ax1.grid()
ax2.grid()

# %%
# ### The results
# We can see that in this case,
# the simulations of the calibrated model are equal to the observations.
# Of course,
# we are in a special configuration where we know the shape of the grey box.
# In practice,
# we could use a parametric model of $\omega$,
# e.g. a machine learning model,
# whose parameters would be calibrated.
#
# ## Initial conditions
# Lastly,
# we may want to calibrate the initial conditions of the simulator,
# such as the initial position which would be incorrect
# (but the ODE model of $\omega$ is correct):
wrong_p_0 = -0.3
simulator = Oscillator()
simulator.rhs_discipline.default_input_data["position"] = wrong_p_0
# %%
# ### The methodology
# For that,
# we suppose that the initial position belongs to the interval $[-2,2]$:
search_space = DesignSpace()
search_space.add_variable(
    "initial_position", lower_bound=-2.0, upper_bound=2.0, value=wrong_p_0
)
# %%
# and we generate a short signal with the reference oscillator:
times = linspace(t_0, t_0 + delta, num=n_sub_nodes)
calibrated_simulator = Oscillator()
simulator_signal = simulator.generate(
    times, {"omega": omega_0, "position": p_0, "velocity": v_0}
)
# %%
# Then,
# we calibrate the simulator:
calibration = CalibrationScenario(
    SignalGeneratorDiscipline(
        # The signal generator
        calibrated_simulator,
        # The state variable names
        ["omega", "position", "velocity"],
        # The parameter names
        [],
        # The observable names
        [output_name],
        # The times of interest
        times,
    ),
    # The input variable names in the calibration data
    ["initial_omega", "initial_velocity"],
    CalibrationMetricSettings(output_name=output_name, metric_name="MSE"),
    # The possible values of a
    search_space,
)
calibration.execute(
    algo_name="NLOPT_COBYLA",
    reference_data={
        "initial_omega": atleast_2d(omega_0),
        "initial_velocity": atleast_2d(v_0),
        output_name: atleast_2d(simulator_signal.evolution[output_name]),
    },
    max_iter=100,
)
# %%
# ### The results
# Finally, we plot the results:
p_0_opt = calibration.optimization_result.x_opt_as_dict["initial_position"][0]
times = linspace(t_0, t_n, num=n_nodes)
simulator_signal = simulator.generate(
    times, {"omega": omega_0, "position": wrong_p_0, "velocity": v_0}
)
calibrated_simulator_signal = calibrated_simulator.generate(
    times, {"omega": omega_0, "position": p_0_opt, "velocity": v_0}
)
plt.plot(
    data.times, data.evolution["position"], "-", color="tab:blue", label="Oscillator"
)
plt.plot(
    simulator_signal.times,
    simulator_signal.evolution["position"],
    "--",
    color="tab:orange",
    label="Simulator",
)
plt.plot(
    calibrated_simulator_signal.times,
    calibrated_simulator_signal.evolution["position"],
    "-.",
    color="tab:green",
    label="Calibrated simulator",
)
plt.grid()
plt.legend()
