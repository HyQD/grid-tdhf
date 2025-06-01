import importlib


def setup_laser(inputs):
    laser_class = inputs.laser
    gauge = inputs.gauge

    params = dict(vars(inputs))

    module = importlib.import_module("grid_tdhf.lasers")
    Laser = getattr(module, laser_class)

    if gauge not in Laser.supported_gauges:
        raise ValueError(f"{laser_class} does not support gauge '{gauge}'.")

    missing_params = Laser.required_params - params.keys() - {"gauge"}
    if missing_params:
        raise ValueError(
            f"Missing required parameters for {laser_class}: {', '.join(sorted(missing_params))}"
        )

    laser_args = {k: params[k] for k in Laser.required_params if k != "gauge"}

    return Laser(**laser_args, gauge=gauge)
