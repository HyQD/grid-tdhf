import tqdm

import numpy as np

from opt_einsum import contract

REQUIRED_IMAG_TIME_PROPAGATION_PARAMS = {
    "u",
    "integrator",
    "rhs",
    "potential_computer",
    "properties_computer",
    "dt",
    "n_orbs",
    "m_list",
    "has_positron",
    "weights",
    "max_iter",
    "conv_tol",
}


def run_imag_time_propagation(
    u,
    integrator,
    rhs,
    potential_computer,
    properties_computer,
    dt,
    n_orbs,
    m_list,
    has_positron,
    weights,
    max_iter,
    conv_tol,
):
    potential_computer.set_state(u)
    potential_computer.compute_direct_potential()
    potential_computer.compute_exchange_potential(u)

    old_energy, old_orbital_energies = properties_computer.compute_energy(u)

    print("Initial energy:", old_energy.real)
    print("initial_orbital_energies:", old_orbital_energies.real)

    for i in range(max_iter):
        u = integrator(u, 0, dt, rhs)

        u = orthonormalize_set(u, m_list, has_positron, weights)

        potential_computer.set_state(u)
        potential_computer.compute_direct_potential()
        potential_computer.compute_exchange_potential(u)

        new_energy, new_orbital_energies = properties_computer.compute_energy(u)

        delta_energy = new_energy - old_energy

        print(i)
        print("delta_energy:", delta_energy.real)
        print("Energy:", new_energy.real)
        print("Orbital energies:", new_orbital_energies.real)

        if abs(delta_energy) < conv_tol:
            break

        old_energy = new_energy

    return u


def orthonormalize_set(u, m_list, has_positron, weights):
    u_new = np.zeros_like(u, dtype=np.complex128)

    if has_positron:
        u_new[-1] = u[-1]
        norm = compute_overlap0(u_new[-1], u_new[-1], weights)
        u_new[-1] /= np.sqrt(norm)
        n_orbs = u.shape[0] - 1
    else:
        n_orbs = u.shape[0]

    for i in range(n_orbs):
        u_new[i] = u[i]
        for j in range(i):
            if m_list[i] != m_list[j]:
                continue
            u_new[i] -= compute_overlap0(u_new[j], u[i], weights) * u_new[j]

        norm = compute_overlap0(u_new[i], u_new[i], weights)
        u_new[i] /= np.sqrt(norm)

    return u_new


def compute_overlap0(u1, u2, weights):
    return contract("lr,lr->", weights * u1.conj(), u2)
