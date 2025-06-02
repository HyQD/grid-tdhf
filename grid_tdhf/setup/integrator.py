import importlib


def setup_integrataor(inputs, imaginary=False):
    integrator_class = inputs.integrator
    gauge = inputs.gauge

    params = dict(vars(inputs))

    module = importlib.import_module("grid_tdhf.integrators")
    Integrator = getattr(module, integrator_class)

    missing_params = Integrator.required_params - params.keys()
    if missing_params:
        raise ValueError(
            f"Missing required parameters for {integrator_class}: {', '.join(sorted(missing_params))}"
        )

    integrator_args = {k: params[k] for k in Integrator.required_params}

    return Integrator(**integrator_args, imaginary=imaginary)
