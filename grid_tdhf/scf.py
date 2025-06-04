import numpy as np
from opt_einsum import contract

from collections import defaultdict

from grid_tdhf.properties import compute_norm


from sympy.physics.quantum.cg import Wigner3j


REQUIRED_SCF_PARAMS_notfinishedyet = [
    "H_core_electron",
    "H_core_positron",
    "poisson_inverse",
    "weights",
    "has_positron",
    "l_list",
    "m_list",
    "gaunt_dict",
    "n_orbs",
    "nl",
    "nr",
    "nL",
]


def run_scf_notfinishedyet(
    *,
    H_core_electron,
    H_core_positron,
    poisson_inverse,
    gaunt_dict,
    weights,
    has_positron,
    l_list,
    m_list,
    n_orbs,
    nl,
    nr,
    nL,
    u_in=None,
    n_scf_it=80,
    scf_alpha=0.6,
    verbose=True,
):
    nl_gs = max(l_list) + 1

    N_orbs = n_orbs + 1 if has_positron else n_orbs

    if u_in is None:
        u_in = get_init_guess(
            H_core_electron,
            H_core_positron,
            poisson_inverse,
            weights,
            nr,
            n_orbs,
            N_orbs,
            l_list,
            nl_gs,
            has_positron,
        )

    gm1, gm2 = setup_gaunt_arrays(n_orbs, nl_gs, nL, m_list, gaunt_dict)

    for _ in range(n_scf_it):
        rho_tilde = contract("olr->r", np.abs(u_in[:n_orbs]) ** 2)
        v_H = np.dot(poisson_inverse[0, :, :], rho_tilde)

        Fx = compute_spherical_exchange_potential(
            u_in, poisson_inverse, gm1, gm2, has_positron
        )

        F = H_core_electron[:nl_gs, :, :] + 2 * np.diag(v_H) - Fx[:, :, :]

        u_out, eps = assign_orbitals_and_energies(n_orbs, nl_gs, nr, l_list[:n_orbs], F)

        norm = compute_norm(u_out, weights)
        u_out /= np.sqrt(norm)[:, None, None]

        u_in[:n_orbs] = (1 - scf_alpha) * u_in[:n_orbs] + scf_alpha * u_out

        norm = compute_norm(u_in, weights)
        u_in /= np.sqrt(norm)[:, None, None]

        if verbose:
            print(f"{eps}")

        if has_positron:
            rho_tilde = contract("olr->r", np.abs(u_in[:n_orbs]) ** 2)
            v_H = np.dot(poisson_inverse[0, :, :], rho_tilde)

            u_p_in = u_in[-1, 0]

            eps_p, u_p = np.linalg.eigh(H_core_positron[0, :, :] - 2 * np.diag(v_H))

            u_p_out = u_p[:, 0]

            norm = compute_norm(u_p_out[None, None, :], weights)
            u_p_out /= np.sqrt(norm[0])

            u_p_in = (1 - scf_alpha) * u_p_in + scf_alpha * u_p_out

            norm = compute_norm(u_p_in[None, None, :], weights)
            u_p_in /= np.sqrt(norm[0])

            u_in[-1] = u_p_in

            if verbose:
                print(f"{eps_p[0]}")

    u_ = np.zeros((N_orbs, nl, nr), dtype=np.complex128)

    u_[:, :nl_gs, :] = u_in

    return u_


def get_init_guess(
    H_core_electron,
    H_core_positron,
    poisson_inverse,
    weights,
    nr,
    n_orbs,
    N_orbs,
    l_list,
    nl_gs,
    has_positron,
    verbose=True,
):
    u = np.zeros((N_orbs, nl_gs, nr))

    u[:n_orbs], eps = assign_orbitals_and_energies(
        n_orbs, nl_gs, nr, l_list[:n_orbs], H_core_electron
    )

    norm = compute_norm(u[:n_orbs], weights)

    print(norm.shape)

    u[:n_orbs] /= np.sqrt(norm)[:, None, None]

    rho_tilde = contract("olr->r", np.abs(u[:n_orbs]) ** 2)
    v_H = np.dot(poisson_inverse[0, :, :], rho_tilde)

    if has_positron:
        eps_p, u_p = np.linalg.eigh(H_core_positron[0, :, :] - 2 * np.diag(v_H))
        u[-1, 0, :] = u_p[:, 0]

    norm = compute_norm(u, weights)
    u /= np.sqrt(norm)[:, None, None]

    return u


def setup_gaunt_arrays(n_orbs, nl_gs, nL, m_list, g_dict):
    gm1 = np.zeros((n_orbs, nl_gs, nL, nl_gs))
    gm2 = np.zeros((n_orbs, nl_gs, nL, nl_gs))

    for i in range(n_orbs):
        mi = m_list[i]
        gm1[i] = (-1) ** (mi) * g_dict[(0, mi)][:nl_gs, :, :nl_gs]
        gm2[i] = g_dict[(mi, 0)][:nl_gs, :, :nl_gs]

    return gm1, gm2


def compute_spherical_exchange_potential(u, poisson_inverse, gm1, gm2, has_positron):
    u = u[:-1] if has_positron else u

    VGx = (
        4
        * np.pi
        * contract(
            "Ila,Lab,Imb,InLl,ImLn->nab",
            u,
            poisson_inverse,
            u.conj(),
            gm1,
            gm2,
        )
    )

    return VGx


def assign_orbitals_and_energies(n_orbs, nl_gs, nr, l_list, hamiltonian):
    u = np.zeros((n_orbs, nl_gs, nr))
    energies = np.zeros(n_orbs)

    l_indices = get_l_indices(l_list)

    for l, indices in l_indices.items():
        eps, u_n = np.linalg.eigh(hamiltonian[l, :, :])

        for i, orb_ind in enumerate(indices):
            u[orb_ind, l, :] = u_n[:, i]
            energies[orb_ind] = eps[i]

    return u, energies


def get_l_indices(l_list):
    l_indices = defaultdict(list)

    for ind, l in enumerate(l_list):
        l_indices[l].append(ind)

    return l_indices


def coeff(l1, l2, l3):
    wigner = float(Wigner3j(l1, 0, l2, 0, l3, 0).doit())
    return (2 * l2 + 1) * wigner**2


def compute_radial_norm(u, weights):
    return np.sum(weights * u.conj() * u)


#####
def run_scf_he(
    H_core_electron,
    poisson_inverse,
    weights,
    nr,
    nl,
    u_in=None,
    scf_n_it=20,
    scf_alpha=0.6,
    verbose=True,
    return_v_H=False,
):
    if u_in is None:
        u_in = init_guess_he(H_core_electron, weights, nr, verbose=verbose)

    u_1s_in = u_in[0, :]

    for _ in range(scf_n_it):
        rho_tilde = np.abs(u_1s_in) ** 2
        v_H = np.dot(poisson_inverse[0, :, :], rho_tilde)

        F = H_core_electron[0, :, :] + np.diag(v_H)

        eps_0, u_n0 = np.linalg.eigh(F)

        u_1s_out = u_n0[:, 0] / np.sqrt(compute_radial_norm(u_n0[:, 0], weights)).real

        u_1s_in = (1 - scf_alpha) * u_1s_in + scf_alpha * u_1s_out

        u_1s_in /= np.sqrt(compute_radial_norm(u_1s_in, weights)).real

        if verbose:
            print(f"{eps_0[0]:.12f}")

    u = np.zeros((1, nl, nr), dtype=np.complex128)
    u[0, 0, :] = u_1s_in

    if return_v_H:
        return u, v_H
    else:
        return u


def run_scf_PsH(
    H_core_electron,
    H_core_positron,
    poisson_inverse,
    weights,
    nr,
    nl,
    u_in=None,
    scf_n_it=20,
    scf_alpha=0.6,
    verbose=True,
    return_v_H=False,
    return_v_p=False,
):
    if u_in is None:
        u_in = init_guess_PsH(
            H_core_electron,
            H_core_positron,
            poisson_inverse,
            weights,
            nr,
            verbose=verbose,
        )

    u_1s_in = u_in[0, :]
    u_Ps_in = u_in[-1, :]

    from grid_tdhf.potentials import compute_spherical_direct_potential

    for _ in range(scf_n_it):
        rho_tilde = np.abs(u_1s_in) ** 2
        v_H = np.dot(poisson_inverse[0, :, :], rho_tilde)

        u_temp = np.zeros((2, 3, nr))
        u_temp[0, 2, :] = u_1s_in

        V = compute_spherical_direct_potential(u_temp, poisson_inverse)

        print(np.max(np.abs(v_H - V)))

        rho_Ps = np.abs(u_Ps_in) ** 2
        v_p = np.dot(poisson_inverse[0, :, :], rho_Ps)

        F = H_core_electron[0, :, :] + np.diag(v_H) - np.diag(v_p)

        eps_0, u_n0 = np.linalg.eigh(F)

        u_1s_out = u_n0[:, 0] / np.sqrt(compute_radial_norm(u_n0[:, 0], weights)).real
        u_1s_in = (1 - scf_alpha) * u_1s_in + scf_alpha * u_1s_out
        u_1s_in /= np.sqrt(compute_radial_norm(u_1s_in, weights)).real

        rho_tilde = np.abs(u_1s_in) ** 2
        v_H = np.dot(poisson_inverse[0, :, :], rho_tilde)

        eps_p, u_p = np.linalg.eigh(H_core_positron[0, :, :] - 2 * np.diag(v_H))

        u_Ps_out = u_p[:, 0] / np.sqrt(compute_radial_norm(u_p[:, 0], weights)).real

        u_Ps_in = (1 - scf_alpha) * u_Ps_in + scf_alpha * u_Ps_out
        u_Ps_in /= np.sqrt(compute_radial_norm(u_Ps_in, weights)).real

        if verbose:
            print(f"Orbital energies:  {eps_0[0]:.12f}")
            print("Ps orbital energy:", eps_p[0])

    u = np.zeros((2, nl, nr), dtype=np.complex128)
    u[0, 0, :] = u_1s_in
    u[-1, 0, :] = u_Ps_in

    if return_v_H and return_v_p:
        return u, v_H, v_p
    elif return_v_H:
        return u, v_H
    elif return_v_p:
        return u, v_p
    else:
        return u


def run_scf_be(
    H_core_electron,
    poisson_inverse,
    weights,
    nr,
    nl,
    u_in=None,
    scf_n_it=20,
    scf_alpha=0.8,
    verbose=True,
):
    if u_in is None:
        u_in = init_guess_be(H_core_electron, weights, nr, verbose=verbose)

    u_1s_in = u_in[0, :]
    u_2s_in = u_in[1, :]

    for _ in range(scf_n_it):
        rho_tilde = np.abs(u_1s_in) ** 2 + np.abs(u_2s_in) ** 2
        v_H = np.dot(poisson_inverse[0, :, :], rho_tilde)

        Fx = coeff(0, 0, 0) * (
            np.einsum("i, ij, j->ij", u_1s_in, poisson_inverse[0, :, :], u_1s_in)
            + np.einsum("i, ij, j->ij", u_2s_in, poisson_inverse[0, :, :], u_2s_in)
        )

        F = H_core_electron[0, :, :] + 2 * np.diag(v_H) - Fx

        eps_0, u_n0 = np.linalg.eigh(F)

        u_1s_out = u_n0[:, 0] / np.sqrt(compute_radial_norm(u_n0[:, 0], weights)).real
        u_2s_out = u_n0[:, 1] / np.sqrt(compute_radial_norm(u_n0[:, 1], weights)).real

        u_1s_in = (1 - scf_alpha) * u_1s_in + scf_alpha * u_1s_out
        u_2s_in = (1 - scf_alpha) * u_2s_in + scf_alpha * u_2s_out

        u_1s_in /= np.sqrt(compute_radial_norm(u_1s_in, weights)).real
        u_2s_in /= np.sqrt(compute_radial_norm(u_2s_in, weights)).real

        if verbose:
            print(f"{eps_0[0]:.12f}, {eps_0[1]:.12f}")

    u = np.zeros((2, nl, nr), dtype=np.complex128)
    u[0, 0, :] = u_1s_in
    u[1, 0, :] = u_2s_in

    return u


def run_scf_ne(
    H_core_electron,
    poisson_inverse,
    weights,
    nr,
    nl,
    u_in=None,
    scf_n_it=20,
    scf_alpha=0.8,
    verbose=True,
):
    if u_in is None:
        u_in = init_guess_ne(H_core_electron, weights, nr, verbose=verbose)

    u_1s_in = u_in[0, :]
    u_2s_in = u_in[1, :]
    u_2p_in = u_in[2, :]

    Fx = np.zeros((2, nr, nr), dtype=np.complex128)
    F = np.zeros((2, nr, nr), dtype=np.complex128)

    for _ in range(scf_n_it):
        rho_tilde = (
            np.abs(u_1s_in) ** 2 + np.abs(u_2s_in) ** 2 + 3 * np.abs(u_2p_in) ** 2
        )
        v_H = np.dot(poisson_inverse[0, :, :], rho_tilde)

        Fx[0, :, :] = coeff(0, 0, 0) * (
            np.einsum("i, ij, j->ij", u_1s_in, poisson_inverse[0, :, :], u_1s_in)
            + np.einsum("i, ij, j->ij", u_2s_in, poisson_inverse[0, :, :], u_2s_in)
        )

        Fx[0, :, :] += (
            3
            * coeff(0, 1, 1)
            * np.einsum("i, ij, j->ij", u_2p_in, poisson_inverse[1, :, :], u_2p_in)
        )

        Fx[1, :, :] = (
            3
            * coeff(1, 0, 1)
            * (
                np.einsum("i, ij, j->ij", u_1s_in, poisson_inverse[1, :, :], u_1s_in)
                + np.einsum("i, ij, j->ij", u_2s_in, poisson_inverse[1, :, :], u_2s_in)
            )
        )

        Fx[1, :, :] += coeff(1, 1, 0) * np.einsum(
            "i, ij, j->ij", u_2p_in, poisson_inverse[0, :, :], u_2p_in
        )

        Fx[1, :, :] += (
            5
            * coeff(1, 1, 2)
            * np.einsum("i, ij, j->ij", u_2p_in, poisson_inverse[2, :, :], u_2p_in)
        )

        F[0, :, :] = H_core_electron[0, :, :] + 2 * np.diag(v_H) - Fx[0, :, :]
        F[1, :, :] = H_core_electron[1, :, :] + 2 * np.diag(v_H) - Fx[1, :, :]

        eps_0, u_n0 = np.linalg.eigh(F[0, :, :])
        eps_1, u_n1 = np.linalg.eigh(F[1, :, :])

        u_1s_out = u_n0[:, 0] / np.sqrt(compute_radial_norm(u_n0[:, 0], weights)).real
        u_2s_out = u_n0[:, 1] / np.sqrt(compute_radial_norm(u_n0[:, 1], weights)).real
        u_2p_out = u_n1[:, 0] / np.sqrt(compute_radial_norm(u_n1[:, 0], weights)).real

        u_1s_in = (1 - scf_alpha) * u_1s_in + scf_alpha * u_1s_out
        u_2s_in = (1 - scf_alpha) * u_2s_in + scf_alpha * u_2s_out
        u_2p_in = (1 - scf_alpha) * u_2p_in + scf_alpha * u_2p_out

        u_1s_in = u_1s_in / np.sqrt(compute_radial_norm(u_1s_in, weights)).real
        u_2s_in = u_2s_in / np.sqrt(compute_radial_norm(u_2s_in, weights)).real
        u_2p_in = u_2p_in / np.sqrt(compute_radial_norm(u_2p_in, weights)).real

        if verbose:
            print(f"{eps_0[0]:.10f}, {eps_0[1]:.10f}, {eps_1[0]:.10f}")

    u = np.zeros((5, nl, nr), dtype=np.complex128)
    u[0, 0, :] = u_1s_in
    u[1, 0, :] = u_2s_in
    u[2, 1, :] = u_2p_in
    u[3, 1, :] = u_2p_in
    u[4, 1, :] = u_2p_in

    return u


def run_scf_ar(
    H_core_electron,
    poisson_inverse,
    weights,
    nr,
    nl,
    u_in=None,
    scf_n_it=20,
    scf_alpha=0.8,
    verbose=True,
):
    if u_in is None:
        u_in = init_guess_ar(H_core_electron, weights, nr, verbose=verbose)

    u_1s_in = u_in[0, :]
    u_2s_in = u_in[1, :]
    u_3s_in = u_in[2, :]
    u_2p_in = u_in[3, :]
    u_3p_in = u_in[4, :]

    Fx = np.zeros((2, nr, nr), dtype=np.complex128)
    F = np.zeros((2, nr, nr), dtype=np.complex128)

    for _ in range(scf_n_it):
        rho_tilde = (
            np.abs(u_1s_in) ** 2
            + np.abs(u_2s_in) ** 2
            + np.abs(u_3s_in) ** 2
            + 3 * np.abs(u_2p_in) ** 2
            + 3 * np.abs(u_3p_in) ** 2
        )
        v_H = np.dot(poisson_inverse[0, :, :], rho_tilde)

        Fx[0, :, :] = coeff(0, 0, 0) * (
            np.einsum("i, ij, j->ij", u_1s_in, poisson_inverse[0, :, :], u_1s_in)
            + np.einsum("i, ij, j->ij", u_2s_in, poisson_inverse[0, :, :], u_2s_in)
            + np.einsum("i, ij, j->ij", u_3s_in, poisson_inverse[0, :, :], u_3s_in)
        )

        Fx[0, :, :] += 3 * coeff(0, 1, 1) * np.einsum(
            "i, ij, j->ij", u_2p_in, poisson_inverse[1, :, :], u_2p_in
        ) + 3 * coeff(0, 1, 1) * np.einsum(
            "i, ij, j->ij", u_3p_in, poisson_inverse[1, :, :], u_3p_in
        )

        Fx[1, :, :] = (
            3
            * coeff(1, 0, 1)
            * (
                np.einsum("i, ij, j->ij", u_1s_in, poisson_inverse[1, :, :], u_1s_in)
                + np.einsum("i, ij, j->ij", u_2s_in, poisson_inverse[1, :, :], u_2s_in)
                + np.einsum("i, ij, j->ij", u_3s_in, poisson_inverse[1, :, :], u_3s_in)
            )
        )

        Fx[1, :, :] += coeff(1, 1, 0) * (
            np.einsum("i, ij, j->ij", u_2p_in, poisson_inverse[0, :, :], u_2p_in)
            + np.einsum("i, ij, j->ij", u_3p_in, poisson_inverse[0, :, :], u_3p_in)
        )

        Fx[1, :, :] += (
            5
            * coeff(1, 1, 2)
            * (
                np.einsum("i, ij, j->ij", u_2p_in, poisson_inverse[2, :, :], u_2p_in)
                + np.einsum("i, ij, j->ij", u_3p_in, poisson_inverse[2, :, :], u_3p_in)
            )
        )

        F[0, :, :] = H_core_electron[0, :, :] + 2 * np.diag(v_H) - Fx[0, :, :]
        F[1, :, :] = H_core_electron[1, :, :] + 2 * np.diag(v_H) - Fx[1, :, :]

        eps_0, u_n0 = np.linalg.eigh(F[0, :, :])
        eps_1, u_n1 = np.linalg.eigh(F[1, :, :])

        if verbose:
            print(
                f"{eps_0[0]:.12f}, {eps_0[1]:.12f}, {eps_0[2]:.12f}, {eps_1[0]:.12f}, {eps_1[1]:.12f}"
            )

        u_1s_out = u_n0[:, 0] / np.sqrt(compute_radial_norm(u_n0[:, 0], weights)).real
        u_2s_out = u_n0[:, 1] / np.sqrt(compute_radial_norm(u_n0[:, 1], weights)).real
        u_3s_out = u_n0[:, 2] / np.sqrt(compute_radial_norm(u_n0[:, 2], weights)).real
        u_2p_out = u_n1[:, 0] / np.sqrt(compute_radial_norm(u_n1[:, 0], weights)).real
        u_3p_out = u_n1[:, 1] / np.sqrt(compute_radial_norm(u_n1[:, 1], weights)).real

        u_1s_in = (1 - scf_alpha) * u_1s_in + scf_alpha * u_1s_out
        u_2s_in = (1 - scf_alpha) * u_2s_in + scf_alpha * u_2s_out
        u_3s_in = (1 - scf_alpha) * u_3s_in + scf_alpha * u_3s_out
        u_2p_in = (1 - scf_alpha) * u_2p_in + scf_alpha * u_2p_out
        u_3p_in = (1 - scf_alpha) * u_3p_in + scf_alpha * u_3p_out

        u_1s_in = u_1s_in / np.sqrt(compute_radial_norm(u_1s_in, weights)).real
        u_2s_in = u_2s_in / np.sqrt(compute_radial_norm(u_2s_in, weights)).real
        u_3s_in = u_2s_in / np.sqrt(compute_radial_norm(u_3s_in, weights)).real
        u_2p_in = u_2p_in / np.sqrt(compute_radial_norm(u_2p_in, weights)).real
        u_3p_in = u_2p_in / np.sqrt(compute_radial_norm(u_3p_in, weights)).real

    u = np.zeros((9, nl, nr), dtype=np.complex128)
    u[0, 0, :] = u_1s_in
    u[1, 0, :] = u_2s_in
    u[2, 0, :] = u_3s_in
    u[3, 1, :] = u_2p_in
    u[4, 1, :] = u_2p_in
    u[5, 1, :] = u_2p_in
    u[6, 1, :] = u_3p_in
    u[7, 1, :] = u_3p_in
    u[8, 1, :] = u_3p_in

    return u


def run_scf_PsCl(
    H_core_electron,
    H_core_positron,
    poisson_inverse,
    weights,
    nr,
    nl,
    u_in=None,
    scf_n_it=20,
    scf_alpha=0.8,
    verbose=True,
):
    if u_in is None:
        u_in = init_guess_PsCl(
            H_core_electron,
            H_core_positron,
            poisson_inverse,
            weights,
            nr,
            verbose=verbose,
        )

    u_1s_in = u_in[0, :]
    u_2s_in = u_in[1, :]
    u_3s_in = u_in[2, :]
    u_2p_in = u_in[3, :]
    u_3p_in = u_in[4, :]
    u_Ps_in = u_in[-1, :]

    Fx = np.zeros((2, nr, nr), dtype=np.complex128)
    F = np.zeros((2, nr, nr), dtype=np.complex128)

    for _ in range(scf_n_it):
        rho_tilde = (
            np.abs(u_1s_in) ** 2
            + np.abs(u_2s_in) ** 2
            + np.abs(u_3s_in) ** 2
            + 3 * np.abs(u_2p_in) ** 2
            + 3 * np.abs(u_3p_in) ** 2
        )

        v_H = np.dot(poisson_inverse[0, :, :], rho_tilde)

        rho_Ps = np.abs(u_Ps_in) ** 2
        v_p = np.dot(poisson_inverse[0, :, :], rho_Ps)

        Fx[0, :, :] = coeff(0, 0, 0) * (
            np.einsum("i, ij, j->ij", u_1s_in, poisson_inverse[0, :, :], u_1s_in)
            + np.einsum("i, ij, j->ij", u_2s_in, poisson_inverse[0, :, :], u_2s_in)
            + np.einsum("i, ij, j->ij", u_3s_in, poisson_inverse[0, :, :], u_3s_in)
        )

        Fx[0, :, :] += 3 * coeff(0, 1, 1) * np.einsum(
            "i, ij, j->ij", u_2p_in, poisson_inverse[1, :, :], u_2p_in
        ) + 3 * coeff(0, 1, 1) * np.einsum(
            "i, ij, j->ij", u_3p_in, poisson_inverse[1, :, :], u_3p_in
        )

        Fx[1, :, :] = (
            3
            * coeff(1, 0, 1)
            * (
                np.einsum("i, ij, j->ij", u_1s_in, poisson_inverse[1, :, :], u_1s_in)
                + np.einsum("i, ij, j->ij", u_2s_in, poisson_inverse[1, :, :], u_2s_in)
                + np.einsum("i, ij, j->ij", u_3s_in, poisson_inverse[1, :, :], u_3s_in)
            )
        )

        Fx[1, :, :] += coeff(1, 1, 0) * (
            np.einsum("i, ij, j->ij", u_2p_in, poisson_inverse[0, :, :], u_2p_in)
            + np.einsum("i, ij, j->ij", u_3p_in, poisson_inverse[0, :, :], u_3p_in)
        )

        Fx[1, :, :] += (
            5
            * coeff(1, 1, 2)
            * (
                np.einsum("i, ij, j->ij", u_2p_in, poisson_inverse[2, :, :], u_2p_in)
                + np.einsum("i, ij, j->ij", u_3p_in, poisson_inverse[2, :, :], u_3p_in)
            )
        )

        F[0, :, :] = (
            H_core_electron[0, :, :] + 2 * np.diag(v_H) - Fx[0, :, :] - np.diag(v_p)
        )
        F[1, :, :] = (
            H_core_electron[1, :, :] + 2 * np.diag(v_H) - Fx[1, :, :] - np.diag(v_p)
        )

        eps_0, u_n0 = np.linalg.eigh(F[0, :, :])
        eps_1, u_n1 = np.linalg.eigh(F[1, :, :])

        u_1s_out = u_n0[:, 0] / np.sqrt(compute_radial_norm(u_n0[:, 0], weights)).real
        u_2s_out = u_n0[:, 1] / np.sqrt(compute_radial_norm(u_n0[:, 1], weights)).real
        u_3s_out = u_n0[:, 2] / np.sqrt(compute_radial_norm(u_n0[:, 2], weights)).real
        u_2p_out = u_n1[:, 0] / np.sqrt(compute_radial_norm(u_n1[:, 0], weights)).real
        u_3p_out = u_n1[:, 1] / np.sqrt(compute_radial_norm(u_n1[:, 1], weights)).real

        u_1s_in = (1 - scf_alpha) * u_1s_in + scf_alpha * u_1s_out
        u_2s_in = (1 - scf_alpha) * u_2s_in + scf_alpha * u_2s_out
        u_3s_in = (1 - scf_alpha) * u_3s_in + scf_alpha * u_3s_out
        u_2p_in = (1 - scf_alpha) * u_2p_in + scf_alpha * u_2p_out
        u_3p_in = (1 - scf_alpha) * u_3p_in + scf_alpha * u_3p_out

        u_1s_in /= np.sqrt(compute_radial_norm(u_1s_in, weights)).real
        u_2s_in /= np.sqrt(compute_radial_norm(u_2s_in, weights)).real
        u_3s_in /= np.sqrt(compute_radial_norm(u_3s_in, weights)).real
        u_2p_in /= np.sqrt(compute_radial_norm(u_2p_in, weights)).real
        u_3p_in /= np.sqrt(compute_radial_norm(u_3p_in, weights)).real

        rho_tilde = (
            np.abs(u_1s_in) ** 2
            + np.abs(u_2s_in) ** 2
            + np.abs(u_3s_in) ** 2
            + 3 * np.abs(u_2p_in) ** 2
            + 3 * np.abs(u_3p_in) ** 2
        )

        v_H = np.dot(poisson_inverse[0, :, :], rho_tilde)

        eps_p, u_p = np.linalg.eigh(H_core_positron[0, :, :] - 2 * np.diag(v_H))

        u_Ps_out = u_p[:, 0] / np.sqrt(compute_radial_norm(u_p[:, 0], weights)).real

        u_Ps_in = (1 - scf_alpha) * u_Ps_in + scf_alpha * u_Ps_out

        u_Ps_in /= np.sqrt(compute_radial_norm(u_Ps_in, weights)).real

        if verbose:
            print(
                f"{eps_0[0]:.6f}, {eps_0[1]:.6f}, {eps_0[2]:.6f}, {eps_1[0]:.6f}, {eps_1[1]:.6f}"
            )
            print("E e+:", eps_p[0])

    u = np.zeros((10, nl, nr), dtype=np.complex128)
    u[0, 0, :] = u_1s_in
    u[1, 0, :] = u_2s_in
    u[2, 0, :] = u_3s_in
    u[3, 1, :] = u_2p_in
    u[4, 1, :] = u_2p_in
    u[5, 1, :] = u_2p_in
    u[6, 1, :] = u_3p_in
    u[7, 1, :] = u_3p_in
    u[8, 1, :] = u_3p_in
    u[-1, 0, :] = u_Ps_in

    return u


def init_guess_he(H_core_electron, weights, nr, verbose=True):
    eps_0, u_n0 = np.linalg.eigh(H_core_electron[0, :, :])

    u_1s_in = u_n0[:, 0] / np.sqrt(compute_radial_norm(u_n0[:, 0], weights)).real

    print("-----", u_1s_in[0])

    u_in = np.zeros((1, nr), dtype=np.complex128)
    u_in[0, :] = u_1s_in

    return u_in


def init_guess_PsH(
    H_core_electron, H_core_positron, poisson_inverse, weights, nr, verbose=True
):
    eps_0, u_n0 = np.linalg.eigh(H_core_electron[0, :, :])
    print("Init eigvals:", eps_0[0])

    u_1s_in = u_n0[:, 0] / np.sqrt(compute_radial_norm(u_n0[:, 0], weights)).real

    rho_tilde = np.abs(u_1s_in) ** 2

    v_H = np.dot(poisson_inverse[0, :, :], rho_tilde)

    eps_p, u_p = np.linalg.eigh(H_core_positron[0, :, :] - 2 * np.diag(v_H))

    u_Ps_in = u_p[:, 0] / np.sqrt(compute_radial_norm(u_p[:, 0], weights)).real

    u_in = np.zeros((2, nr), dtype=np.complex128)
    u_in[0, :] = u_1s_in
    u_in[1, :] = u_Ps_in

    return u_in


def init_guess_be(H_core_electron, weights, nr, verbose=True):
    eps_0, u_n0 = np.linalg.eigh(H_core_electron[0, :, :])
    print("Init eigvals:", eps_0[0], eps_0[1])

    u_1s_in = u_n0[:, 0] / np.sqrt(compute_radial_norm(u_n0[:, 0], weights)).real
    u_2s_in = u_n0[:, 1] / np.sqrt(compute_radial_norm(u_n0[:, 1], weights)).real

    u_in = np.zeros((2, nr), dtype=np.complex128)
    u_in[0, :] = u_1s_in
    u_in[1, :] = u_2s_in

    return u_in


def init_guess_ne(H_core_electron, weights, nr, verbose=True):
    eps_0, u_n0 = np.linalg.eigh(H_core_electron[0, :, :])
    eps_1, u_n1 = np.linalg.eigh(H_core_electron[1, :, :])
    print("Init eigvals:", eps_0[0], eps_0[1], eps_1[0])

    u_1s_in = u_n0[:, 0] / np.sqrt(compute_radial_norm(u_n0[:, 0], weights)).real
    u_2s_in = u_n0[:, 1] / np.sqrt(compute_radial_norm(u_n0[:, 1], weights)).real
    u_2p_in = u_n1[:, 0] / np.sqrt(compute_radial_norm(u_n1[:, 0], weights)).real

    u_in = np.zeros((3, nr), dtype=np.complex128)
    u_in[0, :] = u_1s_in
    u_in[1, :] = u_2s_in
    u_in[2, :] = u_2p_in

    return u_in


def init_guess_ar(H_core_electron, weights, nr, verbose=True):
    eps_0, u_n0 = np.linalg.eigh(H_core_electron[0, :, :])
    eps_1, u_n1 = np.linalg.eigh(H_core_electron[1, :, :])
    print("Init eigvals:", eps_0[0], eps_0[1], eps_0[2], eps_1[0], eps_1[1])

    u_1s_in = u_n0[:, 0] / np.sqrt(compute_radial_norm(u_n0[:, 0], weights)).real
    u_2s_in = u_n0[:, 1] / np.sqrt(compute_radial_norm(u_n0[:, 1], weights)).real
    u_3s_in = u_n0[:, 2] / np.sqrt(compute_radial_norm(u_n0[:, 2], weights)).real
    u_2p_in = u_n1[:, 0] / np.sqrt(compute_radial_norm(u_n1[:, 0], weights)).real
    u_3p_in = u_n1[:, 1] / np.sqrt(compute_radial_norm(u_n1[:, 1], weights)).real

    u_in = np.zeros((5, nr), dtype=np.complex128)
    u_in[0, :] = u_1s_in
    u_in[1, :] = u_2s_in
    u_in[2, :] = u_3s_in
    u_in[3, :] = u_2p_in
    u_in[4, :] = u_3p_in

    return u_in


def init_guess_PsCl(
    H_core_electron, H_core_positron, poisson_inverse, weights, nr, verbose=True
):
    eps_0, u_n0 = np.linalg.eigh(H_core_electron[0, :, :])
    eps_1, u_n1 = np.linalg.eigh(H_core_electron[1, :, :])
    print("Init eigvals:", eps_0[0], eps_0[1], eps_0[2], eps_1[0], eps_1[1])

    u_1s_in = u_n0[:, 0] / np.sqrt(compute_radial_norm(u_n0[:, 0], weights)).real
    u_2s_in = u_n0[:, 1] / np.sqrt(compute_radial_norm(u_n0[:, 1], weights)).real
    u_3s_in = u_n0[:, 2] / np.sqrt(compute_radial_norm(u_n0[:, 2], weights)).real
    u_2p_in = u_n1[:, 0] / np.sqrt(compute_radial_norm(u_n1[:, 0], weights)).real
    u_3p_in = u_n1[:, 1] / np.sqrt(compute_radial_norm(u_n1[:, 1], weights)).real

    rho_tilde = (
        np.abs(u_1s_in) ** 2
        + np.abs(u_2s_in) ** 2
        + np.abs(u_3s_in) ** 2
        + 3 * np.abs(u_2p_in) ** 2
        + 3 * np.abs(u_3p_in) ** 2
    )

    v_H = np.dot(poisson_inverse[0, :, :], rho_tilde)

    eps_p, u_p = np.linalg.eigh(H_core_positron[0, :, :] - 2 * np.diag(v_H))

    u_Ps_in = u_p[:, 0] / np.sqrt(compute_radial_norm(u_p[:, 0], weights)).real

    u_in = np.zeros((6, nr), dtype=np.complex128)
    u_in[0, :] = u_1s_in
    u_in[1, :] = u_2s_in
    u_in[2, :] = u_3s_in
    u_in[3, :] = u_2p_in
    u_in[4, :] = u_3p_in
    u_in[5, :] = u_Ps_in

    return u_in


REQUIRED_SCF_PARAMS = [
    "atom",
    "H_core_electron",
    "H_core_positron",
    "poisson_inverse",
    "weights",
    "nr",
    "nl",
]


def run_scf(
    *,
    atom,
    H_core_electron,
    H_core_positron,
    poisson_inverse,
    weights,
    nr,
    nl,
    u_in=None,
    n_scf_iter=60,
    scf_alpha=0.8,
    verbose=True,
):
    atom = atom.lower()
    if atom == "he":
        return run_scf_he(
            H_core_electron,
            poisson_inverse,
            weights,
            nr,
            nl,
            u_in=u_in,
            scf_n_it=n_scf_iter,
            scf_alpha=scf_alpha,
            verbose=verbose,
        )
    elif atom == "be":
        return run_scf_be(
            H_core_electron,
            poisson_inverse,
            weights,
            nr,
            nl,
            u_in=u_in,
            scf_n_it=n_scf_iter,
            scf_alpha=scf_alpha,
            verbose=verbose,
        )
    elif atom == "ne":
        return run_scf_ne(
            H_core_electron,
            poisson_inverse,
            weights,
            nr,
            nl,
            u_in=u_in,
            scf_n_it=n_scf_iter,
            scf_alpha=scf_alpha,
            verbose=verbose,
        )
    elif atom == "ar":
        return run_scf_ar(
            H_core_electron,
            poisson_inverse,
            weights,
            nr,
            nl,
            u_in=u_in,
            scf_n_it=n_scf_iter,
            scf_alpha=scf_alpha,
            verbose=verbose,
        )
    elif atom == "psh":
        return run_scf_PsH(
            H_core_electron,
            H_core_positron,
            poisson_inverse,
            weights,
            nr,
            nl,
            u_in=u_in,
            scf_n_it=n_scf_iter,
            scf_alpha=scf_alpha,
            verbose=verbose,
        )
    elif atom == "psli":
        return run_scf_PsLi(
            H_core_electron,
            H_core_positron,
            poisson_inverse,
            weights,
            nr,
            nl,
            u_in=u_in,
            scf_n_it=n_scf_iter,
            scf_alpha=scf_alpha,
            verbose=verbose,
        )
    elif atom == "psf":
        return run_scf_PsF(
            H_core_electron,
            H_core_positron,
            poisson_inverse,
            weights,
            nr,
            nl,
            u_in=u_in,
            scf_n_it=n_scf_iter,
            scf_alpha=scf_alpha,
            verbose=verbose,
        )
    elif atom == "pscl":
        return run_scf_PsCl(
            H_core_electron,
            H_core_positron,
            poisson_inverse,
            weights,
            nr,
            nl,
            u_in=u_in,
            scf_n_it=n_scf_iter,
            scf_alpha=scf_alpha,
            verbose=verbose,
        )
    else:
        raise NotImplementedError(
            f"SCF GS calculation for has not been implemented for {atom}"
        )
