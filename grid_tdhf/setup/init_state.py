import numpy as np

from grid_tdhf.scf import run_scf, REQUIRED_SCF_PARAMS
from grid_tdhf.time_propagation.imag import (
    run_imag_time_propagation,
    REQUIRED_IMAG_TIME_PROPAGATION_PARAMS,
)

from grid_tdhf.utils import select_keys

from grid_tdhf.setup.integrator import setup_integrator
from grid_tdhf.setup.rhs import setup_rhs
from grid_tdhf.computers.potential_computer import PotentialComputer
from grid_tdhf.computers.properties_computer import PropertiesComputer


def setup_init_state(inputs, system_info, angular_arrays, radial_arrays, aux_arrays):
    init_state = inputs.init_state

    params = {
        **vars(inputs),
        **vars(system_info),
        **vars(angular_arrays),
        **vars(radial_arrays),
        **vars(aux_arrays),
    }

    if init_state.lower() == "scf":
        return run_scf(**select_keys(params, REQUIRED_SCF_PARAMS))

    elif init_state.lower() == "itp":
        integrator = setup_integrator(
            inputs, system_info, radial_arrays, imaginary=True
        )
        potential_computer = PotentialComputer(
            inputs, system_info, radial_arrays, aux_arrays
        )
        properties_computer = PropertiesComputer(
            system_info, angular_arrays, radial_arrays, aux_arrays, potential_computer
        )

        rhs = setup_rhs(
            inputs,
            system_info,
            angular_arrays,
            radial_arrays,
            aux_arrays,
            potential_computer,
        )

        # temporary hack
        u = run_scf(
            **select_keys(params, REQUIRED_SCF_PARAMS, exclude={"n_scf_iter"}),
            n_scf_iter=0,
        )

        simulation_params = select_keys(params, REQUIRED_IMAG_TIME_PROPAGATION_PARAMS)

        return run_imag_time_propagation(
            **simulation_params,
            u=u,
            integrator=integrator,
            rhs=rhs,
            potential_computer=potential_computer,
            properties_computer=properties_computer,
        )

    else:
        return load_state(inputs.init_state_file)


def load_state(init_state_file):
    return np.load(init_state_file)
