def get_imag_time_overrides(u, system_config):
    m_max = system_config.m_max

    l_max = m_max
    nl = l_max + 1
    nL = 2 * nl

    centrifugal_potential_l = system_config.centrifugal_potential_l[:nl]

    poisson_inverse = system_config.poisson_inverse[:nL, :, :]

    gaunt_dict = {}

    for m1 in range(-m_max, m_max + 1):
        for m2 in range(-m_max, m_max + 1):
            gaunt_dict[(m1, m2)] = system_config.gaunt_dict[(m1, m2)][:nl, :nL, :nl]

    overrides = {
        "l_max": l_max,
        "nl": nl,
        "nL": nL,
        "u": u[:, :nl, :].copy(),
        "centrifugal_potential_l": centrifugal_potential_l,
        "poisson_inverse": poisson_inverse,
        "gaunt_dict": gaunt_dict,
        "integrator_name": system_config.itp_integrator_name,
    }

    return overrides
