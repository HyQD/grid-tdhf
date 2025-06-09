from mpi4py import MPI

import tqdm

REQUIRED_TIME_PROPAGATION_PARAMS = {
    "comm",
    "u",
    "integrator",
    "rhs",
    "potential_computer",
    "dt",
    "total_steps",
    "init_step",
    "t0",
    "mask",
    "sampler",
    "checkpoint_manager",
}


def run_time_propagation(
    comm,
    u,
    integrator,
    rhs,
    potential_computer,
    dt,
    total_steps,
    init_step=0,
    t0=0,
    mask=None,
    sampler=None,
    checkpoint_manager=None,
):
    rank = comm.Get_rank()

    global_u = u
    t = t0

    local_u = global_u[rank : rank + 1, :, :]
    potential_computer.set_state(global_u)
    potential_computer.compute_direct_potential()
    potential_computer.compute_exchange_potential(local_u)

    sampler.sample(local_u, t, 0)

    if init_step == 0 and rank == 0:
        checkpoint_manager.checkpoint(u, t, 0)

    for i in range(init_step, total_steps):
        potential_computer.set_state(global_u)
        potential_computer.compute_direct_potential()

        local_u = global_u[rank : rank + 1, :, :]

        local_u = integrator(local_u, t, dt, rhs)

        if mask is not None:
            local_u = mask * local_u

        comm.Allgather([local_u, MPI.COMPLEX16], [global_u, MPI.COMPLEX16])

        t += dt

        sampler.sample(local_u, t, i + 1)

        if rank == 0:
            checkpoint_manager.checkpoint(global_u, t, i + 1)
            print(f"{i} / {total_steps}")

    if rank == 0:
        checkpoint_manager.finalize(global_u, t, i)
