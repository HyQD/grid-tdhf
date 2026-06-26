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


def setup_imp_cdm1(
    runtime_config,
    imaginary=False,
    used_inputs=None,
    param_mapping=None,
    potential_computer=None,
):
    from grid_tdhf.integrators import IMPCDM1

    preconditioner = setup_preconditioner(
        runtime_config,
        imaginary=imaginary,
        used_inputs=used_inputs,
        param_mapping=param_mapping,
    )

    args = {**vars(runtime_config)}
    integrator_args = resolve_required_params(
        IMPCDM1.required_params, args, used_inputs, param_mapping
    )

    integrator = IMPCDM1(
        **integrator_args, imaginary=imaginary, preconditioner=preconditioner
    )

    return integrator


def setup_imp_cdm2(
    runtime_config,
    imaginary=False,
    used_inputs=None,
    param_mapping=None,
    potential_computer=None,
):
    from grid_tdhf.integrators import IMPCDM2

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
        IMPCDM2.required_params, args, used_inputs, param_mapping
    )

    integrator = IMPCDM2(
        **integrator_args,
        imaginary=imaginary,
        preconditioner=preconditioner,
        potential_computer=potential_computer
    )

    return integrator


SETUP_DISPATCH = {
    "IMP-CDM1": setup_imp_cdm1,
    "IMP-CDM2": setup_imp_cdm2,
}
