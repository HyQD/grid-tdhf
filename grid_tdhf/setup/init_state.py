import numpy as np

from grid_tdhf.scf import run_scf, REQUIRED_SCF_KEYS
from grid_tdhf.time_propagation.imag import run_imaginary_time_propagation

from grid_tdhf.utils import select_keys


def setup_init_state(inputs, system_info, aux_arrays, gll, rme):
    init_state = inputs.init_state

    if init_state.lower() == "scf":
        args = {
            **dict(vars(inputs)),
            **dict(vars(system_info)),
            **dict(vars(aux_arrays)),
            **dict(vars(gll)),
            **dict(vars(rme)),
        }

        return run_scf(**select_keys(args, REQUIRED_SCF_KEYS))
    elif init_state.lower() == "itp":
        return run_imaginary_time_propagation(inputs.start_guess)
    else:
        return load_state(inputs.init_state_file)


def load_state(init_state_file):
    return np.load(init_state_file)
