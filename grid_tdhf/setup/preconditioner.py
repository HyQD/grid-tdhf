import importlib


def setup_preconditioner(inputs, system_info, radial_arrays, imaginary=False):
    preconditioner_name = inputs.preconditioner

    params = {**vars(inputs), **vars(system_info), **vars(radial_arrays)}

    module = importlib.import_module("grid_tdhf.preconditioners")
    Preconditioner = getattr(module, preconditioner_name)

    missing_params = Preconditioner.required_params - params.keys() - {"imaginary"}
    if missing_params:
        raise ValueError(
            f"Missing required parameters for {preconditioner_name}: {', '.join(sorted(missing_params))}"
        )

    preconditioner_args = {k: params[k] for k in Preconditioner.required_params if k in params}

    return Preconditioner(**preconditioner_args, imaginary=imaginary)
