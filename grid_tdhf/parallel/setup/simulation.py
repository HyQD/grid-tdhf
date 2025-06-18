import numpy as np
from types import SimpleNamespace

from grid_tdhf.utils import resolve_required_params
from grid_tdhf.parallel.time_propagation.real import (
    run_time_propagation,
    REQUIRED_TIME_PROPAGATION_PARAMS,
)

from grid_tdhf.constants import LASER_GROUPS


def run_simulation(
    integrator,
    rhs,
    mask,
    potential_computer,
    sampler,
    checkpoint_manager,
    simulation_config,
    simulation_info,
    used_inputs=None,
    param_mapping=None,
):
    params = {**vars(simulation_config), **vars(simulation_info)}

    simulation_params = resolve_required_params(
        REQUIRED_TIME_PROPAGATION_PARAMS, params, used_inputs, param_mapping
    )

    run_time_propagation(
        **simulation_params,
        integrator=integrator,
        rhs=rhs,
        mask=mask,
        potential_computer=potential_computer,
        sampler=sampler,
        checkpoint_manager=checkpoint_manager,
    )


def setup_simulation(simulation_config):
    total_time = determine_simulation_time(simulation_config)
    total_steps = int(total_time / simulation_config.dt)

    return SimpleNamespace(
        init_step=0,
        total_steps=total_steps,
        t0=simulation_config.t0,
    )


def determine_simulation_time(simulation_config):
    if simulation_config.laser_name in LASER_GROUPS.SINUSOIDAL:
        ncycles = simulation_config.ncycles
        ncycles_after_pulse = simulation_config.ncycles_after_pulse
        omega = simulation_config.omega

        T = 2 * np.pi * (ncycles + ncycles_after_pulse) / omega

    elif simulation_config.laser_name in LASER_GROUPS.DELTA_PULSE:
        T = simulation_config.total_time

    return T
