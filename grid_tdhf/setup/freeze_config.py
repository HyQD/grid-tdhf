from grid_tdhf.potentials import compute_spherical_direct_potential


def generate_freeze_config(u, inputs, aux_arrays):
    overrides = {}

    if inputs.frozen_electrons:
        poisson_inverse = aux_arrays.poisson_inverse
        coulomb_potential = aux_arrays.coulomb_potential

        V_d_electron = compute_spherical_direct_potential(u[:-1], poisson_inverse)

        overrides.update(
            {
                "n_orbs": 0,
                "N_orbs": 1,
                "coulomb_potential": coulomb_potential + 2 * V_d_electron,
                "u": u[-1:],
                "m_list": [0],
                "m_max": 0,
            }
        )

    return overrides
