import numpy as np


from grid_tdhf.scf import run_scf, REQUIRED_SCF_PARAMS
from grid_tdhf.time_propagation.imag import (
    run_imag_time_propagation,
    REQUIRED_IMAG_TIME_PROPAGATION_PARAMS,
)

from grid_tdhf.utils import resolve_required_params

from grid_tdhf.setup.integrator import setup_integrator
from grid_tdhf.setup.rhs import setup_rhs
from grid_tdhf.computers.potential_computer import PotentialComputer
from grid_tdhf.computers.properties_computer import PropertiesComputer
from grid_tdhf.setup.load_run import load_state
from grid_tdhf.config.system import generate_runtime_config
from grid_tdhf.config.imag_time import get_imag_time_overrides
from grid_tdhf.setup.init_guess import generate_init_guess


def setup_init_state(system_config, used_inputs=None):
    load_run = system_config.load_run
    init_state = system_config.init_state

    if load_run is not None:
        init_u, _, _ = load_state(load_run)

    elif init_state.lower() == "scf":
        init_u = setup_scf(system_config, used_inputs)

    elif init_state.lower() == "itp":
        init_u = setup_imag_time_propagation(system_config, used_inputs)

    else:
        init_u = load_state(system_config.init_state_file)

    if system_config.save_gs:
        save_gs(system_config, init_u)

    N_orbs = system_config.N_orbs
    nl = system_config.nl
    nr = system_config.nr

    u = np.zeros((N_orbs, nl, nr), dtype=np.complex128)
    u[: init_u.shape[0], : init_u.shape[1], : init_u.shape[2]] = init_u

    return u


def setup_scf(system_config, used_inputs=None):
    param_mapping = {
        "n_iter": "n_scf_iter",
        "alpha": "scf_alpha",
        "conv_tol": "scf_conv_tol",
        "max_iter": "scf_max_iter",
    }

    u = generate_init_guess(system_config)

    params = {**vars(system_config)}
    scf_params = resolve_required_params(
        REQUIRED_SCF_PARAMS, params, used_inputs, param_mapping
    )

    return run_scf(**scf_params, u=u)


def setup_imag_time_propagation(system_config, used_inputs=None):
    param_mapping = {
        "dt": "itp_dt",
        "conv_tol": "itp_conv_tol",
        "max_iter": "itp_max_iter",
    }

    u = generate_init_guess(system_config)

    overrides = get_imag_time_overrides(u, system_config)
    imag_time_config = generate_runtime_config(system_config, overrides)

    potential_computer = PotentialComputer(imag_time_config)
    properties_computer = PropertiesComputer(imag_time_config, potential_computer)

    integrator = setup_integrator(
        imag_time_config,
        potential_computer,
        imaginary=True,
        used_inputs=used_inputs,
        param_mapping=param_mapping,
    )

    rhs = setup_rhs(imag_time_config, potential_computer)

    params = {**vars(imag_time_config)}
    imag_time_params = resolve_required_params(
        REQUIRED_IMAG_TIME_PROPAGATION_PARAMS, params, used_inputs, param_mapping
    )

    return run_imag_time_propagation(
        **imag_time_params,
        integrator=integrator,
        rhs=rhs,
        potential_computer=potential_computer,
        properties_computer=properties_computer,
    )


def save_gs(system_config, u):
    import uuid

    fileroot = str(uuid.uuid4())

    inputs = {
        "atom": system_config.atom,
        "N": system_config.N,
        "r_max": system_config.r_max,
        "init_state": system_config.init_state,
        "itp_dt": system_config.itp_dt,
        "itp_conv_tol": system_config.itp_conv_tol,
    }

    info = {"inputs": inputs}

    np.save(f"output_gs/{fileroot}_gs", u)
    np.savez(f"output_gs/{fileroot}_info", **info)
