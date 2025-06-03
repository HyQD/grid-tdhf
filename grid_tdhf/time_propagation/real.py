import tqdm

REQUIRED_TIME_PROPAGATION_PARAMS = {
    "u",
    "integrator",
    "rhs",
    "potential_computer",
    "dt",
    "total_steps",
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
    t0=0,
    i_init=0,
    mask=None,
    sampler=None,
    checkpoint_manager=None,
):

    t = t0

    potential_computer.set_state(u)
    potential_computer.compute_direct_potential()
    potential_computer.compute_exchange_potential(u)

    energy, orb_energies = sampler.properties_computer.compute_energy(u)
    print("energy", energy)
    print(orb_energies)

    for i in tqdm.tqdm(range(i_init, total_steps)):
        sampler.sample(u, t, i)
        checkpoint_manager.checkpoint(i, u, t)

        potential_computer.set_state(u)
        potential_computer.compute_direct_potential()

        u = integrator(u, t, dt, rhs)

        if mask is not None:
            u = mask * u

        t += dt

    checkpoint_manager.finalize(i, u, t)
