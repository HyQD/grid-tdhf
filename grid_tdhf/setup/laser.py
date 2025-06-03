import importlib

from grid_tdhf.utils import select_keys


def setup_laser(inputs):
    laser_name = inputs.laser_name
    gauge = inputs.gauge

    params = {**vars(inputs)}

    module = importlib.import_module("grid_tdhf.lasers")
    Laser = getattr(module, laser_name)

    if gauge not in Laser.supported_gauges:
        raise ValueError(f"{laser_name} does not support gauge '{gauge}'.")

    missing_params = Laser.required_params - params.keys() - {"gauge"}
    if missing_params:
        raise ValueError(
            f"Missing required parameters for {laser_name}: {', '.join(sorted(missing_params))}"
        )

    laser_args = select_keys(params, Laser.required_params, exclude={"gauge"})

    return Laser(**laser_args, gauge=gauge)
