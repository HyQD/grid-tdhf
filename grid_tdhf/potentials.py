import numpy as np
from opt_einsum import contract


def compute_spherical_direct_potential(u, poisson_inverse):
    rho = contract("olr->r", np.abs(u) ** 2)
    return np.dot(poisson_inverse[0, :, :], rho)


def compute_direct_potential(
    u_bar, n_orbs, nl, nr, m_list, m_max, poisson_inverse, gaunt_dict, has_positron
):
    V_d_electron = {
        m: np.zeros((nl, nl, nr), dtype=np.complex128) for m in range(-m_max, m_max + 1)
    }

    if has_positron:
        V_d_positron = {
            m: np.zeros((nl, nl, nr), dtype=np.complex128)
            for m in range(-m_max, m_max + 1)
        }
    else:
        V_d_positron = None

    for mp in range(-m_max, m_max + 1):
        g_mat2 = gaunt_dict[(mp, mp)]
        for j in range(n_orbs):
            mj = m_list[j]
            g_mat1 = gaunt_dict[(mj, mj)]
            V_d_electron[mp] += (
                4
                * np.pi
                * contract(
                    "Lrs,ms,ns,mLn,oLl->olr",
                    poisson_inverse,
                    u_bar[j].conj(),
                    u_bar[j],
                    g_mat1,
                    g_mat2,
                )
            )

        if has_positron:
            g_mat = gaunt_dict[(0, 0)]
            V_d_positron[mp] = (
                4
                * np.pi
                * contract(
                    "Lrs,ms,ns,mLn,oLl->olr",
                    poisson_inverse,
                    u_bar[-1].conj(),
                    u_bar[-1],
                    g_mat,
                    g_mat2,
                )
            )

    return V_d_electron, V_d_positron


def compute_exchange_potential(
    u_bar, u_tilde, n_orbs, nl, nr, m_list, poisson_inverse, gaunt_dict
):
    V_x = np.zeros((n_orbs, n_orbs, nl, nl, nr), dtype=np.complex128)

    for j in range(n_orbs):
        for p in range(n_orbs):
            mj = m_list[j]
            mp = m_list[p]
            g_mat1 = gaunt_dict[(mj, mp)]
            g_mat2 = gaunt_dict[(mp, mj)]
            V_x[j, p, :, :, :] = (
                4
                * (-1) ** (mj - mp)
                * np.pi
                * contract(
                    "Lsr, mr, nr, mLn, oLl -> ols",
                    poisson_inverse,
                    u_bar[j].conj(),
                    u_tilde[p],
                    g_mat1,
                    g_mat2,
                )
            )

    return V_x
