import importlib


def setup_preconditioner(inputs, imaginary=False):
    preconditioner_class = inputs.preconditioner
    gauge = inputs.gauge

    params = dict(vars(inputs))

    module = importlib.import_module("grid_tdhf.preconditioners")
    Preconditioner = getattr(module, preconditioner_class)

    missing_params = Preconditioner.required_params - params.keys()
    if missing_params:
        raise ValueError(
            f"Missing required parameters for {preconditioner_class}: {', '.join(sorted(missing_params))}"
        )

    preconditioner_args = {k: params[k] for k in Preconditioner.required_params}

    return Preconditioner(**preconditioner_args, imaginary=imaginary)
