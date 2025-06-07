import numpy as np
from sympy.physics.quantum.cg import Wigner3j


def coeff(l1, l2, l3):
    wigner = float(Wigner3j(l1, 0, l2, 0, l3, 0).doit())
    return (2 * l2 + 1) * wigner**2


def compute_radial_norm(u, weights):
    return np.sum(weights * u.conj() * u)


def run_scf_he(
    u,
    H_core_electron,
    poisson_inverse,
    weights,
    nr,
    nl,
    n_iter=20,
    alpha=0.6,
    verbose=True,
    return_v_H=False,
):
    u_1s_in = u[0, 0, :]

    for _ in range(n_iter):
        rho_tilde = np.abs(u_1s_in) ** 2
        v_H = np.dot(poisson_inverse[0, :, :], rho_tilde)

        F = H_core_electron[0, :, :] + np.diag(v_H)

        eps_0, u_n0 = np.linalg.eigh(F)

        u_1s_out = u_n0[:, 0] / np.sqrt(compute_radial_norm(u_n0[:, 0], weights)).real

        u_1s_in = (1 - alpha) * u_1s_in + alpha * u_1s_out

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
    u,
    H_core_electron,
    H_core_positron,
    poisson_inverse,
    weights,
    nr,
    nl,
    n_iter=20,
    alpha=0.6,
    verbose=True,
    return_v_H=False,
    return_v_p=False,
):
    u_1s_in = u[0, 0, :]
    u_Ps_in = u[-1, 0, :]

    from grid_tdhf.potentials import compute_spherical_direct_potential

    for _ in range(n_iter):
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
        u_1s_in = (1 - alpha) * u_1s_in + alpha * u_1s_out
        u_1s_in /= np.sqrt(compute_radial_norm(u_1s_in, weights)).real

        rho_tilde = np.abs(u_1s_in) ** 2
        v_H = np.dot(poisson_inverse[0, :, :], rho_tilde)

        eps_p, u_p = np.linalg.eigh(H_core_positron[0, :, :] - 2 * np.diag(v_H))

        u_Ps_out = u_p[:, 0] / np.sqrt(compute_radial_norm(u_p[:, 0], weights)).real

        u_Ps_in = (1 - alpha) * u_Ps_in + alpha * u_Ps_out
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
    u,
    H_core_electron,
    poisson_inverse,
    weights,
    nr,
    nl,
    n_iter=20,
    alpha=0.8,
    verbose=True,
):
    u_1s_in = u[0, 0, :]
    u_2s_in = u[1, 0, :]

    for _ in range(n_iter):
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

        u_1s_in = (1 - alpha) * u_1s_in + alpha * u_1s_out
        u_2s_in = (1 - alpha) * u_2s_in + alpha * u_2s_out

        u_1s_in /= np.sqrt(compute_radial_norm(u_1s_in, weights)).real
        u_2s_in /= np.sqrt(compute_radial_norm(u_2s_in, weights)).real

        if verbose:
            print(f"{eps_0[0]:.12f}, {eps_0[1]:.12f}")

    u = np.zeros((2, nl, nr), dtype=np.complex128)
    u[0, 0, :] = u_1s_in
    u[1, 0, :] = u_2s_in

    return u


def run_scf_ne(
    u,
    H_core_electron,
    poisson_inverse,
    weights,
    nr,
    nl,
    n_iter=20,
    alpha=0.8,
    verbose=True,
):
    u_1s_in = u[0, 0, :]
    u_2s_in = u[1, 0, :]
    u_2p_in = u[3, 1, :]

    Fx = np.zeros((2, nr, nr), dtype=np.complex128)
    F = np.zeros((2, nr, nr), dtype=np.complex128)

    for _ in range(n_iter):
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

        u_1s_in = (1 - alpha) * u_1s_in + alpha * u_1s_out
        u_2s_in = (1 - alpha) * u_2s_in + alpha * u_2s_out
        u_2p_in = (1 - alpha) * u_2p_in + alpha * u_2p_out

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
    u,
    H_core_electron,
    poisson_inverse,
    weights,
    nr,
    nl,
    n_iter=20,
    alpha=0.8,
    verbose=True,
):
    u_1s_in = u[0, 0, :]
    u_2s_in = u[1, 0, :]
    u_3s_in = u[2, 0, :]
    u_2p_in = u[4, 1, :]
    u_3p_in = u[7, 1, :]

    Fx = np.zeros((2, nr, nr), dtype=np.complex128)
    F = np.zeros((2, nr, nr), dtype=np.complex128)

    for _ in range(n_iter):
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

        u_1s_in = (1 - alpha) * u_1s_in + alpha * u_1s_out
        u_2s_in = (1 - alpha) * u_2s_in + alpha * u_2s_out
        u_3s_in = (1 - alpha) * u_3s_in + alpha * u_3s_out
        u_2p_in = (1 - alpha) * u_2p_in + alpha * u_2p_out
        u_3p_in = (1 - alpha) * u_3p_in + alpha * u_3p_out

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
    u,
    H_core_electron,
    H_core_positron,
    poisson_inverse,
    weights,
    nr,
    nl,
    n_iter=20,
    alpha=0.8,
    verbose=True,
):
    u_1s_in = u[0, 0, :]
    u_2s_in = u[1, 0, :]
    u_3s_in = u[2, 0, :]
    u_2p_in = u[4, 1, :]
    u_3p_in = u[7, 1, :]
    u_Ps_in = u[-1, 0, :]

    Fx = np.zeros((2, nr, nr), dtype=np.complex128)
    F = np.zeros((2, nr, nr), dtype=np.complex128)

    for _ in range(n_iter):
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

        u_1s_in = (1 - alpha) * u_1s_in + alpha * u_1s_out
        u_2s_in = (1 - alpha) * u_2s_in + alpha * u_2s_out
        u_3s_in = (1 - alpha) * u_3s_in + alpha * u_3s_out
        u_2p_in = (1 - alpha) * u_2p_in + alpha * u_2p_out
        u_3p_in = (1 - alpha) * u_3p_in + alpha * u_3p_out

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

        u_Ps_in = (1 - alpha) * u_Ps_in + alpha * u_Ps_out

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


REQUIRED_SCF_PARAMS = [
    "u",
    "atom",
    "H_core_electron",
    "H_core_positron",
    "poisson_inverse",
    "weights",
    "nr",
    "nl",
    "n_iter",
    "alpha",
    "verbose",
]


def run_scf(
    *,
    u,
    atom,
    H_core_electron,
    H_core_positron,
    poisson_inverse,
    weights,
    nr,
    nl,
    n_iter=60,
    alpha=0.8,
    verbose=True,
):
    atom = atom.lower()
    if atom == "he":
        return run_scf_he(
            u,
            H_core_electron,
            poisson_inverse,
            weights,
            nr,
            nl,
            n_iter=n_iter,
            alpha=alpha,
            verbose=verbose,
        )
    elif atom == "be":
        return run_scf_be(
            u,
            H_core_electron,
            poisson_inverse,
            weights,
            nr,
            nl,
            n_iter=n_iter,
            alpha=alpha,
            verbose=verbose,
        )
    elif atom == "ne":
        return run_scf_ne(
            u,
            H_core_electron,
            poisson_inverse,
            weights,
            nr,
            nl,
            n_iter=n_iter,
            alpha=alpha,
            verbose=verbose,
        )
    elif atom == "ar":
        return run_scf_ar(
            u,
            H_core_electron,
            poisson_inverse,
            weights,
            nr,
            nl,
            n_iter=n_iter,
            alpha=alpha,
            verbose=verbose,
        )
    elif atom == "psh":
        return run_scf_PsH(
            u,
            H_core_electron,
            H_core_positron,
            poisson_inverse,
            weights,
            nr,
            nl,
            n_iter=n_iter,
            alpha=alpha,
            verbose=verbose,
        )
    elif atom == "psli":
        return run_scf_PsLi(
            u,
            H_core_electron,
            H_core_positron,
            poisson_inverse,
            weights,
            nr,
            nl,
            n_iter=n_iter,
            alpha=alpha,
            verbose=verbose,
        )
    elif atom == "psf":
        return run_scf_PsF(
            u,
            H_core_electron,
            H_core_positron,
            poisson_inverse,
            weights,
            nr,
            nl,
            n_iter=n_iter,
            alpha=alpha,
            verbose=verbose,
        )
    elif atom == "pscl":
        return run_scf_PsCl(
            u,
            H_core_electron,
            H_core_positron,
            poisson_inverse,
            weights,
            nr,
            nl,
            n_iter=n_iter,
            alpha=alpha,
            verbose=verbose,
        )
    else:
        raise NotImplementedError(
            f"SCF GS calculation for has not been implemented for {atom}"
        )
