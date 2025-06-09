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
    comm,
    u,
    V_d_electron,
    V_d_positron,
    V_x,
    *,
    n_orbs,
    m_list,
    has_positron,
    is_positron,
    coulomb_potential,
    centrifugal_potential_l,
    centrifugal_potential_r,
    D2,
    weights,
    single_orbital=False,
):
    rank = comm.Get_rank()

    nl = u.shape[1]
    nr = u.shape[2]

    u_1 = np.zeros((nl, nr), dtype=np.complex128)
    u_2 = np.zeros((nl, nr), dtype=np.complex128)
    u_3 = np.zeros((nl, nr), dtype=np.complex128)

    if not is_positron:
        m = m_list[0]

        u_1 += contract("Ij, ij->Ii", u[rank], -(1 / 2) * D2)
        u_1 += contract("Ik, k->Ik", u[rank], coulomb_potential)
        u_temp = contract("I, Ii->Ii", centrifugal_potential_l, u[rank])
        u_1 += contract("i, Ii->Ii", centrifugal_potential_r, u_temp)

        if single_orbital:
            u_2 += contract("ijr,jr->ir", V_d_electron[m], u[rank])
        else:
            u_2 += 2 * contract("ijr,jr->ir", V_d_electron[m], u[rank])

            for j in range(n_orbs):
                u_2 -= contract("ijr,jr->ir", V_x[j], u[j])

        if has_positron:
            m = m_list[0]
            u_3 -= contract("ijr,jr->ir", V_d_positron[m], u[rank])

    else:
        u_2 += contract("Ij, ij->Ii", u[rank], -(1 / 2) * D2)
        u_2 -= contract("Ik, k->Ik", u[rank], coulomb_potential)
        u_temp = contract("I, Ii->Ii", centrifugal_potential_l, u[rank])
        u_2 += contract("i, Ii->Ii", centrifugal_potential_r, u_temp)

        u_2 -= 2 * contract("ijr,jr->ir", V_d_electron[0], u[rank])

    HF_energy = contract("lr, lr ->", weights * u[rank].conj(), 2 * u_1 + u_2)
    orbital_energies = contract("lr, lr -> ", weights * u[rank].conj(), u_1 + u_2 + u_3)

    return HF_energy, orbital_energies
