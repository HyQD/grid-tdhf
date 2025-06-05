from types import SimpleNamespace
import numpy as np


def generate_simulation_config(
    u, inputs, system_info, angular_arrays, radial_arrays, aux_arrays, overrides={}
):
    simulation_config = {
        "u": u,
        "full_state": u,
        "active_orbitals": np.ones(u.shape[0], dtype=bool),
        **vars(inputs),
        **vars(system_info),
        **vars(angular_arrays),
        **vars(radial_arrays),
        **vars(aux_arrays),
        **overrides,
    }

    return SimpleNamespace(**simulation_config)
