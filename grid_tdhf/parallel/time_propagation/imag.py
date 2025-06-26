from mpi4py import MPI

import numpy as np
from opt_einsum import contract

REQUIRED_IMAG_TIME_PROPAGATION_PARAMS = {
    "comm",
    "u",
    "integrator",
    "rhs",
    "potential_computer",
    "properties_computer",
    "dt",
    "m_list_tot",
    "has_positron",
    "weights",
    "max_iter",
    "conv_tol",
}


def run_imag_time_propagation(
    comm,
    u,
    integrator,
    rhs,
    potential_computer,
    properties_computer,
    dt,
    m_list_tot,
    has_positron,
    weights,
    max_iter,
    conv_tol,
):
    rank = comm.Get_rank()
    size = comm.Get_size()

    global_u = u

    local_u = global_u[rank : rank + 1, :, :]
    potential_computer.set_state(global_u)
    potential_computer.compute_direct_potential()
    potential_computer.compute_exchange_potential(local_u)

    local_energy, local_orbital_energy = properties_computer.compute_energy(
        comm, global_u
    )
    old_total_energy = comm.allreduce(local_energy, op=MPI.SUM)

    if rank == 0:
        old_orbital_energies = np.empty(size, dtype=local_orbital_energy.dtype)
        new_orbital_energies = np.empty(size, dtype=local_orbital_energy.dtype)
    else:
        old_orbital_energies = None
        new_orbital_energies = None

    comm.Gather(local_orbital_energy, old_orbital_energies, root=0)

    if rank == 0:
        print("Initial energy:", old_total_energy.real)
        print("initial_orbital_energies:", old_orbital_energies.real)

    for i in range(max_iter):
        local_u = global_u[rank : rank + 1, :, :]

        local_u = integrator(local_u, 0, dt, rhs)

        comm.Allgather([local_u, MPI.COMPLEX16], [global_u, MPI.COMPLEX16])

        global_u = orthonormalize_set(global_u, m_list_tot, has_positron, weights)

        local_u = global_u[rank : rank + 1, :, :]
        potential_computer.set_state(global_u)
        potential_computer.compute_direct_potential()
        potential_computer.compute_exchange_potential(local_u)

        local_energy, local_orbital_energy = properties_computer.compute_energy(
            comm, global_u
        )

        new_total_energy = comm.allreduce(local_energy, op=MPI.SUM)
        comm.Gather(local_orbital_energy, new_orbital_energies, root=0)

        delta_energy = new_total_energy - old_total_energy

        if rank == 0:
            print(i)
            print("delta_energy:", delta_energy.real)
            print("Energy:", new_total_energy.real)
            print("Orbital energies:", new_orbital_energies.real)

        if abs(delta_energy) < conv_tol:
            break

        old_total_energy = new_total_energy

    return global_u


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
