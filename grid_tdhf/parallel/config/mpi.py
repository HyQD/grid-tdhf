from grid_tdhf.potentials import compute_spherical_direct_potential

import numpy as np


def get_mpi_overrides(comm, system_config):
    size = comm.Get_size()
    rank = comm.Get_rank()

    overrides = {}

    m_list = system_config.m_list
    has_positron = system_config.has_positron
    is_positron = True if rank == size - 1 and has_positron else False

    n_orbs_tot = system_config.n_orbs
    N_orbs_tot = system_config.N_orbs

    overrides.update(
        {
            "n_orbs": 1 if not is_positron else 0,
            "N_orbs": 1,
            "n_orbs_tot": n_orbs_tot,
            "N_orbs_tot": N_orbs_tot,
            "is_positron": is_positron,
            "m_list": [m_list[rank]],
            "m_list_tot": m_list,
            "active_orbitals": np.ones(size, dtype=bool),
            "comm": comm,
        }
    )

    return overrides
