import importlib

from grid_tdhf.utils import resolve_required_params


def setup_laser(simulation_config, used_inputs=None, param_mapping=None):
    laser_name = simulation_config.laser_name

    params = {**vars(simulation_config)}

    module = importlib.import_module("grid_tdhf.lasers")
    Laser = getattr(module, laser_name)

    laser_args = resolve_required_params(
        Laser.required_params, params, used_inputs, param_mapping
    )

    missing_params = Laser.required_params - laser_args.keys()
    if missing_params:
        raise ValueError(
            f"Missing required parameters for {laser_name}: {', '.join(sorted(missing_params))}"
        )

    return Laser(**laser_args)
