from dataclasses import dataclass
import numpy as np
from sympy.physics.quantum.cg import Wigner3j


@dataclass
class AuxiliaryArrays:
    H_core_electron: np.ndarray
    H_core_positron: np.ndarray
    poisson_inverse: np.ndarray
    gaunt_dict: np.ndarray
    coulomb_potential: np.ndarray
    centrifugal_potential_l: np.ndarray
    centrifugal_potential_r: np.ndarray


def setup_auxiliary_arrays(inputs, system_info, gll, rme):
    r_max = inputs.r_max
    nl = inputs.nl
    nL = inputs.nL

    m_max = system_info.m_max
    Z = system_info.Z

    r = rme.r
    nr = rme.nr
    r_dot = rme.r_dot
    D2 = rme.D2
    T_D2 = -(1 / 2) * D2

    PN_x = gll.PN_x
    weights = gll.weights

    l_nums = np.arange(nl)

    centrifugal_potential_l = l_nums * (l_nums + 1)
    centrifugal_potential_r = 1 / (2 * r**2)
    coulomb_potential = -Z / r

    poisson_inverse = setup_poisson_inverse(D2, r, r_dot, PN_x, nL, nr)
    poisson_inverse += setup_poisson_inverse_boundary(r, nL, nr, r_max, weights)

    H_core_electron = setup_H_core(
        T_D2,
        centrifugal_potential_l,
        centrifugal_potential_r,
        coulomb_potential,
        nl,
        nr,
    )
    H_core_positron = setup_H_core(
        T_D2,
        centrifugal_potential_l,
        centrifugal_potential_r,
        coulomb_potential,
        nl,
        nr,
        positron=True,
    )

    gaunt_dict = setup_gaunt_dict(nl, nL, m_max)

    return AuxiliaryArrays(
        H_core_electron=H_core_electron,
        H_core_positron=H_core_positron,
        poisson_inverse=poisson_inverse,
        gaunt_dict=gaunt_dict,
        coulomb_potential=coulomb_potential,
        centrifugal_potential_l=centrifugal_potential_l,
        centrifugal_potential_r=centrifugal_potential_r,
    )


def setup_H_core(
    T_D2,
    centrifugal_potential_l,
    centrifugal_potential_r,
    coulomb_potential,
    nl,
    nr,
    positron=False,
):
    if positron:
        sign = -1
    else:
        sign = 1

    H_core = np.zeros((nl, nr, nr))

    for i in range(nl):
        H_core[i, :, :] = (
            T_D2
            + centrifugal_potential_l[i] * np.diag(centrifugal_potential_r)
            + sign * np.diag(coulomb_potential)
        )

    return H_core


def setup_poisson_inverse(D2, r, r_dot, PN_x, nL, nr):
    A = np.zeros((nL, nr, nr))
    Ainv = np.zeros((nL, nr, nr))
    calF = np.zeros((nL, nr, nr))

    for l2 in range(nL):
        A[l2, :, :] = np.einsum("i, ij, j->ij", r, D2, r) - l2 * (l2 + 1) * np.eye(nr)
        Ainv[l2, :, :] = np.linalg.inv(A[l2, :, :])
        calF[l2, :, :] = -np.einsum(
            "i, ij, j->ij",
            PN_x / np.sqrt(r_dot),
            Ainv[l2, :, :],
            PN_x / np.sqrt(r_dot),
        )

    return calF


def setup_poisson_inverse_boundary(r, nL, nr, r_max, weights):
    G = np.zeros((nL, nr, nr))

    for L in range(nL):
        for i in range(nr):
            G[L, i, :] = (
                (1 / (2 * L + 1)) * (weights / r_max ** (2 * L + 1)) * r[i] ** L * r**L
            )

    return G


def setup_gaunt_dict(nl, nL, m_max):
    g_dict = {}

    for m1 in range(-m_max, m_max + 1):
        for m2 in range(-m_max, m_max + 1):
            g_mat = np.zeros((nl, nL, nl))
            for l1 in range(nl):
                for L in range(nL):
                    for l2 in range(nl):
                        if abs(m1) > l1 or abs(m2) > l2:
                            continue
                        if abs(l1 - l2) > L or abs(l1 + l2) < L:
                            continue
                        g_mat[l1, L, l2] = (-1) ** m1 * gaunt(
                            l1, -m1, L, (m1 - m2), l2, m2
                        )

            g_dict[(m1, m2)] = g_mat

    return g_dict


def gaunt(l1, m1, l2, m2, l3, m3):
    T1 = np.sqrt(((2 * l1 + 1) * (2 * l2 + 1) * (2 * l3 + 1)) / (4 * np.pi))
    T2 = float(Wigner3j(l1, 0, l2, 0, l3, 0).doit())
    T3 = float(Wigner3j(l1, m1, l2, m2, l3, m3).doit())
    return T1 * T2 * T3
