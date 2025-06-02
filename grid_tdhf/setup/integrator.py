import importlib

from grid_tdhf.setup.preconditioner import setup_preconditioner


def setup_integrator(inputs, system_info, radial_arrays, imaginary=False):
    integrator_name = inputs.integrator

    params = {**vars(inputs), **vars(system_info), **vars(radial_arrays)}

    module = importlib.import_module("grid_tdhf.integrators")
    Integrator = getattr(module, integrator_name)

    if "preconditioner_obj" in Integrator.required_params:
        preconditioner_obj = setup_preconditioner(inputs, system_info, radial_arrays, imaginary=imaginary)
        params["preconditioner_obj"] = preconditioner_obj

    missing_params = Integrator.required_params - params.keys() - {"imaginary"}
    if missing_params:
        raise ValueError(
            f"Missing required parameters for {integrator_name}: {', '.join(sorted(missing_params))}"
        )

    integrator_args = {k: params[k] for k in Integrator.required_params if k in params}

    return Integrator(**integrator_args, imaginary=imaginary)
