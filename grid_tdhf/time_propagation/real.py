import tqdm

REQUIRED_TIME_PROPAGATION_PARAMS = {
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

    t = t0

    potential_computer.set_state(u)
    potential_computer.compute_direct_potential()
    potential_computer.compute_exchange_potential(u)

    if init_step == 0:
        sampler.sample(u, t, 0)
        checkpoint_manager.checkpoint(u, t, 0)

    for i in tqdm.tqdm(range(init_step, total_steps)):
        potential_computer.set_state(u)
        potential_computer.compute_direct_potential()

        u = integrator(u, t, dt, rhs)

        if mask is not None:
            u = mask * u

        t += dt

        sampler.sample(u, t, i + 1)
        checkpoint_manager.checkpoint(u, t, i + 1)

    checkpoint_manager.finalize(u, t, i)
