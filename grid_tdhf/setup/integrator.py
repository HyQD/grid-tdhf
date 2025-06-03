import importlib

from grid_tdhf.utils import select_keys
from grid_tdhf.setup.preconditioner import setup_preconditioner


def setup_integrator(inputs, system_info, radial_arrays, imaginary=False):
    integrator_name = inputs.integrator_name

    params = {**vars(inputs), **vars(system_info), **vars(radial_arrays)}

    if inputs.frozen_electrons:
        params["N_orbs"] = 1
    elif inputs.frozen_positron:
        params["N_orbs"] = params["n_orbs"]
        params["has_positron"] = False

    module = importlib.import_module("grid_tdhf.integrators")
    Integrator = getattr(module, integrator_name)

    if "preconditioner" in Integrator.required_params:
        preconditioner = setup_preconditioner(
            inputs, system_info, radial_arrays, imaginary=imaginary
        )
        params["preconditioner"] = preconditioner

    missing_params = Integrator.required_params - params.keys() - {"imaginary"}
    if missing_params:
        raise ValueError(
            f"Missing required parameters for {integrator_name}: {', '.join(sorted(missing_params))}"
        )

    integrator_args = select_keys(params, Integrator.required_params)

    return Integrator(**integrator_args, imaginary=imaginary)
