from types import SimpleNamespace


def generate_simulation_config(
    u, inputs, system_info, angular_arrays, radial_arrays, aux_arrays, overrides={}
):
    simulation_config = {
        "u": u,
        **vars(inputs),
        **vars(system_info),
        **vars(angular_arrays),
        **vars(radial_arrays),
        **vars(aux_arrays),
        **overrides,
    }

    return SimpleNamespace(**simulation_config)
