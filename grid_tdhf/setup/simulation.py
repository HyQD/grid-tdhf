import numpy as np
from types import SimpleNamespace

from grid_tdhf.utils import select_keys
from grid_tdhf.time_propagation.real import (
    run_time_propagation,
    REQUIRED_TIME_PROPAGATION_PARAMS,
)

from grid_tdhf.constants import LASER_GROUPS


def run_simulation(
    u,
    integrator,
    rhs,
    mask,
    potential_computer,
    sampler,
    checkpoint_manager,
    inputs,
    system_info,
    radial_arrays,
    simulation_info,
):
    params = {
        **vars(inputs),
        **vars(system_info),
        **vars(radial_arrays),
        **vars(simulation_info),
    }

    simulation_params = select_keys(params, REQUIRED_TIME_PROPAGATION_PARAMS)

    run_time_propagation(
        **simulation_params,
        u=u,
        integrator=integrator,
        rhs=rhs,
        mask=mask,
        potential_computer=potential_computer,
        sampler=sampler,
        checkpoint_manager=checkpoint_manager,
    )


def setup_simulation(inputs):
    total_time = determine_simulation_time(inputs)
    total_steps = int(total_time / inputs.dt)

    return SimpleNamespace(
        total_steps=total_steps,
    )


def determine_simulation_time(inputs):
    if inputs.laser_name in LASER_GROUPS.SINUSOIDAL:
        T = 2 * np.pi * (inputs.ncycles + inputs.ncycles_after_pulse) / inputs.omega
    elif inputs.laser_name in LASER_GROUPS.DELTA_PULSE:
        T = inputs.total_time

    return T
