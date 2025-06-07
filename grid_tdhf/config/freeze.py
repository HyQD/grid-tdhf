from grid_tdhf.potentials import compute_spherical_direct_potential

import numpy as np


def get_freeze_overrides(u, system_config):
    overrides = {}

    if system_config.frozen_electrons:
        poisson_inverse = system_config.poisson_inverse
        coulomb_potential = system_config.coulomb_potential

        V_d_electron = compute_spherical_direct_potential(u[:-1], poisson_inverse)

        active_orbitals = np.zeros(u.shape[0], dtype=bool)
        active_orbitals[-1] = True

        overrides.update(
            {
                "n_orbs": 0,
                "N_orbs": 1,
                "m_list": [0],
                "m_max": 0,
                "u": u[-1:].copy(),
                "full_state": u,
                "active_orbitals": active_orbitals,
                "coulomb_potential": coulomb_potential + 2 * V_d_electron,
            }
        )

    elif system_config.frozen_positron:
        poisson_inverse = system_config.poisson_inverse
        coulomb_potential = system_config.coulomb_potential

        V_d_positron = compute_spherical_direct_potential(u[-1:], poisson_inverse)

        active_orbitals = np.ones(u.shape[0], dtype=bool)
        active_orbitals[-1] = False

        overrides.update(
            {
                "has_positron": False,
                "N_orbs": system_config.n_orbs,
                "m_list": system_config.m_list[:-1],
                "u": u[:-1].copy(),
                "full_state": u,
                "active_orbitals": active_orbitals,
                "coulomb_potential": coulomb_potential - V_d_positron,
            }
        )

    return overrides
