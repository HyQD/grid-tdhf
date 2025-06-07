import numpy as np


def init_guess_he(H_core_electron, weights, nr, verbose=True):
    eps_0, u_n0 = np.linalg.eigh(H_core_electron[0, :, :])
    print("Init eigval:", eps_0[0])

    u_1s_in = u_n0[:, 0] / np.sqrt(compute_radial_norm(u_n0[:, 0], weights)).real

    u_in = np.zeros((1, 1, nr), dtype=np.complex128)
    u_in[0, 0, :] = u_1s_in

    return u_in


def init_guess_PsH(
    H_core_electron, H_core_positron, poisson_inverse, weights, nr, verbose=True
):
    eps_0, u_n0 = np.linalg.eigh(H_core_electron[0, :, :])

    u_1s_in = u_n0[:, 0] / np.sqrt(compute_radial_norm(u_n0[:, 0], weights)).real

    rho_tilde = np.abs(u_1s_in) ** 2

    v_H = np.dot(poisson_inverse[0, :, :], rho_tilde)

    eps_p, u_p = np.linalg.eigh(H_core_positron[0, :, :] - 2 * np.diag(v_H))
    print("Init eigvals:", eps_0[0], eps_p[0])

    u_Ps_in = u_p[:, 0] / np.sqrt(compute_radial_norm(u_p[:, 0], weights)).real

    u_in = np.zeros((2, 1, nr), dtype=np.complex128)
    u_in[0, 0, :] = u_1s_in
    u_in[1, 0, :] = u_Ps_in

    return u_in


def init_guess_be(H_core_electron, weights, nr, verbose=True):
    eps_0, u_n0 = np.linalg.eigh(H_core_electron[0, :, :])
    print("Init eigvals:", eps_0[0], eps_0[1])

    u_1s_in = u_n0[:, 0] / np.sqrt(compute_radial_norm(u_n0[:, 0], weights)).real
    u_2s_in = u_n0[:, 1] / np.sqrt(compute_radial_norm(u_n0[:, 1], weights)).real

    u_in = np.zeros((2, 1, nr), dtype=np.complex128)
    u_in[0, 0, :] = u_1s_in
    u_in[1, 0, :] = u_2s_in

    return u_in


def init_guess_ne(H_core_electron, weights, nr, verbose=True):
    eps_0, u_n0 = np.linalg.eigh(H_core_electron[0, :, :])
    eps_1, u_n1 = np.linalg.eigh(H_core_electron[1, :, :])
    print("Init eigvals:", eps_0[0], eps_0[1], eps_1[0])

    u_1s_in = u_n0[:, 0] / np.sqrt(compute_radial_norm(u_n0[:, 0], weights)).real
    u_2s_in = u_n0[:, 1] / np.sqrt(compute_radial_norm(u_n0[:, 1], weights)).real
    u_2p_in = u_n1[:, 0] / np.sqrt(compute_radial_norm(u_n1[:, 0], weights)).real

    u_in = np.zeros((5, 2, nr), dtype=np.complex128)
    u_in[0, 0, :] = u_1s_in
    u_in[1, 0, :] = u_2s_in
    u_in[2, 1, :] = u_2p_in
    u_in[3, 1, :] = u_2p_in
    u_in[4, 1, :] = u_2p_in

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

    u_in = np.zeros((9, 2, nr), dtype=np.complex128)
    u_in[0, 0, :] = u_1s_in
    u_in[1, 0, :] = u_2s_in
    u_in[2, 0, :] = u_3s_in
    u_in[3, 1, :] = u_2p_in
    u_in[4, 1, :] = u_2p_in
    u_in[5, 1, :] = u_2p_in
    u_in[6, 1, :] = u_3p_in
    u_in[7, 1, :] = u_3p_in
    u_in[8, 1, :] = u_3p_in

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

    u_in = np.zeros((10, 2, nr), dtype=np.complex128)
    u_in[0, 0, :] = u_1s_in
    u_in[1, 0, :] = u_2s_in
    u_in[2, 0, :] = u_3s_in
    u_in[3, 1, :] = u_2p_in
    u_in[4, 1, :] = u_2p_in
    u_in[5, 1, :] = u_2p_in
    u_in[6, 1, :] = u_3p_in
    u_in[7, 1, :] = u_3p_in
    u_in[8, 1, :] = u_3p_in
    u_in[9, 0, :] = u_Ps_in

    return u_in


def compute_radial_norm(u, weights):
    return np.sum(weights * u.conj() * u)


def generate_init_guess(system_config):
    H_core_electron = system_config.H_core_electron
    H_core_positron = system_config.H_core_positron
    poisson_inverse = system_config.poisson_inverse
    weights = system_config.weights
    nr = system_config.nr
    atom = system_config.atom.lower()

    if atom == "he":
        u = init_guess_he(H_core_electron, weights, nr)
    elif atom == "be":
        u = init_guess_be(H_core_electron, weights, nr)
    elif atom == "ne":
        u = init_guess_ne(H_core_electron, weights, nr)
    elif atom == "ar":
        u = init_guess_ar(H_core_electron, weights, nr)
    elif atom == "psh":
        u = init_guess_PsH(
            H_core_electron, H_core_positron, poisson_inverse, weights, nr
        )
    elif atom == "pscl":
        u = init_guess_PsCl(
            H_core_electron, H_core_positron, poisson_inverse, weights, nr
        )

    return u
