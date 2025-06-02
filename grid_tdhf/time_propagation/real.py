import tqdm

REQUIRED_TIME_PROPAGATION_PARAMS = {
    "u",
    "integrator",
    "potential_computer",
    "rhs",
    "dt",
    "n_it",
    "t0",
    "i_ini",
    "mask",
    "sampler",
    "checkpoint_manager",
}


def run_time_propagation(
    u,
    integrator,
    potential_computer,
    rhs,
    dt,
    n_it,
    t0=0,
    i_init=0,
    mask=None,
    sampler=None,
    checkpoint_manager=None,
):

    t = t0

    for i in tqdm.tqdm(range(i_init, n_it)):
        potential_computer.set_state(u)
        potential_computer.construct_direct_potential()

        u = integrator(u, t, dt, rhs)

        if mask is not None:
            u = mask * u

        t += dt
