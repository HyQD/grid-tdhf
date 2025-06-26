from grid_tdhf.utils import resolve_required_params
from grid_tdhf.setup.preconditioner import setup_preconditioner

from grid_tdhf.config.system import generate_runtime_config


def setup_integrator(
    runtime_config,
    potential_computer,
    imaginary=False,
    used_inputs=None,
    param_mapping=None,
):
    integrator_name = runtime_config.integrator_name

    integrator_setup_fn = SETUP_DISPATCH[integrator_name]

    return integrator_setup_fn(
        runtime_config,
        imaginary=imaginary,
        used_inputs=used_inputs,
        param_mapping=param_mapping,
        potential_computer=potential_computer,
    )


def setup_cn(
    runtime_config,
    imaginary=False,
    used_inputs=None,
    param_mapping=None,
    potential_computer=None,
):
    from grid_tdhf.parallel.integrators import CN

    preconditioner = setup_preconditioner(
        runtime_config,
        imaginary=imaginary,
        used_inputs=used_inputs,
        param_mapping=param_mapping,
    )

    args = {**vars(runtime_config)}
    integrator_args = resolve_required_params(
        CN.required_params, args, used_inputs, param_mapping
    )

    integrator = CN(
        **integrator_args, imaginary=imaginary, preconditioner=preconditioner
    )

    return integrator


def setup_cncmf2(
    runtime_config,
    imaginary=False,
    used_inputs=None,
    param_mapping=None,
    potential_computer=None,
):
    from grid_tdhf.parallel.integrators import CNCMF2

    half_dt_config = generate_runtime_config(
        runtime_config, {"dt": runtime_config.dt / 2}
    )

    preconditioner = setup_preconditioner(
        half_dt_config,
        imaginary=imaginary,
        used_inputs=used_inputs,
        param_mapping=param_mapping,
    )

    args = {**vars(runtime_config)}
    integrator_args = resolve_required_params(
        CNCMF2.required_params, args, used_inputs, param_mapping
    )

    integrator = CNCMF2(
        **integrator_args,
        imaginary=imaginary,
        preconditioner=preconditioner,
        potential_computer=potential_computer
    )

    return integrator


SETUP_DISPATCH = {
    "CN": setup_cn,
    "CNCMF2": setup_cncmf2,
}
