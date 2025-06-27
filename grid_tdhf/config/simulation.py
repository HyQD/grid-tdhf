import numpy as np


def get_simulation_overrides(u, system_config):
    N_full_state_orbs = u.shape[0]

    simulation_config = {
        "u": u,
        "full_state": u,
        "active_orbitals": np.ones(N_full_state_orbs, dtype=bool),
    }

    return simulation_config
