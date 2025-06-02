import numpy as np

from grid_tdhf.scf import run_scf, REQUIRED_SCF_PARAMS
from grid_tdhf.time_propagation.imag import run_imag_time_propagation

from grid_tdhf.utils import select_keys


def setup_init_state(inputs, system_info, radial_arrays, aux_arrays):
    init_state = inputs.init_state

    if init_state.lower() == "scf":
        args = {
            **vars(inputs),
            **vars(system_info),
            **vars(radial_arrays),
            **vars(aux_arrays),
        }

        return run_scf(**select_keys(args, REQUIRED_SCF_PARAMS))
    elif init_state.lower() == "itp":
        return run_imag_time_propagation(inputs.start_guess)
    else:
        return load_state(inputs.init_state_file)


def load_state(init_state_file):
    return np.load(init_state_file)
