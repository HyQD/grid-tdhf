import numpy as np


def get_simulation_overrides(u, system_config):
    N_orbs = system_config.N_orbs

    simulation_config = {
        "u": u,
        "full_state": u,
        "active_orbitals": np.ones(N_orbs, dtype=bool),
    }

    return simulation_config
