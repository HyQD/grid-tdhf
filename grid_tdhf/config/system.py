from types import SimpleNamespace


def generate_system_config(
    inputs, system_info, angular_arrays, radial_arrays, aux_arrays
):
    return SimpleNamespace(
        **{
            **vars(inputs),
            **vars(system_info),
            **vars(angular_arrays),
            **vars(radial_arrays),
            **vars(aux_arrays),
        }
    )


def generate_runtime_config(system_config, overrides):
    return SimpleNamespace(**{**vars(system_config), **overrides})
