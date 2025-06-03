import numpy as np
from opt_einsum import contract

from grid_methods.spherical_coordinates.utils import quadrature


def compute_norm(u, weights):
    return contract("plr,plr->p", weights * u.conj(), u)


def compute_overlap(u1, u2, weights):
    return contract("plr,plr->p", weights * u1.conj(), u2)


def compute_expec_z(u, z_Omega, r, weights, m_list):
    n_orbs = u.shape[0]
    u_temp = np.zeros_like(u, dtype=np.complex128)

    for p in range(n_orbs):
        m = m_list[p]
        u_temp[p] = contract("ij, jr -> ir", z_Omega[m], u[p])

    return contract("pir,r,pir->p", weights * u.conj(), r, u_temp)


def compute_energy(
    u,
    V_d_electron,
    V_d_positron,
    V_x,
    *,
    n_orbs,
    m_list,
    has_positron,
    coulomb_potential,
    centrifugal_potential_l,
    centrifugal_potential_r,
    D2,
    weights,
    single_orbital=False,
):
    u_1 = np.zeros_like(u, dtype=np.complex128)
    u_2 = np.zeros_like(u, dtype=np.complex128)
    u_3 = np.zeros_like(u, dtype=np.complex128)

    for i in range(n_orbs):
        m = m_list[i]

        u_1[i] += contract("Ij, ij->Ii", u[i], -(1 / 2) * D2)
        u_1[i] += contract("Ik, k->Ik", u[i], coulomb_potential)
        u_temp = contract("I, Ii->Ii", centrifugal_potential_l, u[i])
        u_1[i] += contract("i, Ii->Ii", centrifugal_potential_r, u_temp)

        if single_orbital:
            u_2[i] += contract("ijr,jr->ir", V_d_electron[m], u[i])
        else:
            u_2[i] += 2 * contract("ijr,jr->ir", V_d_electron[m], u[i])

            for j in range(n_orbs):
                u_2[i] -= contract("ijr,jr->ir", V_x[j, i], u[j])

    if has_positron:

        for i in range(n_orbs):
            m = m_list[i]
            u_3[i] -= contract("ijr,jr->ir", V_d_positron[m], u[i])

        u_2[-1] += contract("Ij, ij->Ii", u[-1], -(1 / 2) * D2)
        u_2[-1] -= contract("Ik, k->Ik", u[-1], coulomb_potential)
        u_temp = contract("I, Ii->Ii", centrifugal_potential_l, u[-1])
        u_2[-1] += contract("i, Ii->Ii", centrifugal_potential_r, u_temp)

        u_2[-1] -= 2 * contract("ijr,jr->ir", V_d_electron[0], u[-1])

    HF_energy = contract("plr, plr ->", weights * u.conj(), 2 * u_1 + u_2)
    orbital_energies = contract("plr, plr -> p", weights * u.conj(), u_1 + u_2 + u_3)

    return HF_energy, orbital_energies
