import importlib

from grid_tdhf.utils import select_keys
from grid_tdhf.setup.preconditioner import setup_preconditioner


def setup_integrator(simulation_config, imaginary=False):
    integrator_name = simulation_config.integrator_name

    params = {**vars(simulation_config)}

    module = importlib.import_module("grid_tdhf.integrators")
    Integrator = getattr(module, integrator_name)

    if "preconditioner" in Integrator.required_params:
        preconditioner = setup_preconditioner(simulation_config, imaginary=imaginary)
        params["preconditioner"] = preconditioner

    missing_params = Integrator.required_params - params.keys() - {"imaginary"}
    if missing_params:
        raise ValueError(
            f"Missing required parameters for {integrator_name}: {', '.join(sorted(missing_params))}"
        )

    integrator_args = select_keys(params, Integrator.required_params)

    return Integrator(**integrator_args, imaginary=imaginary)
