from grid_tdhf.utils import select_keys


def setup_rhs(
    inputs,
    system_info,
    angular_matrices,
    radial_arrays,
    aux_arrays,
    laser,
    potential_computer,
):

    args = {
        **vars(inputs),
        **vars(system_info),
        **vars(angular_matrices),
        **vars(radial_arrays),
        **vars(aux_arrays),
    }

    from grid_tdhf.rhs import CompositeRHS

    from grid_tdhf.rhs import RHSCore
    from grid_tdhf.rhs import RHSMeanField

    if inputs.gauge == "velocity":
        from grid_tdhf.rhs import RHSDipoleVelocityGauge as RHSDipole
    elif inputs.gauge == "length":
        from grid_tdhf.rhs import RHSDipoleLengthGauge as RHSDipole

    core = RHSCore(**select_keys(args, RHSCore.required_params))
    mean_field = RHSMeanField(
        **select_keys(args, RHSMeanField.required_params),
        potential_computer=potential_computer
    )
    dipole = RHSDipole(**select_keys(args, RHSDipole.required_params), laser=laser)

    rhs = CompositeRHS(core=core, mean_field=mean_field, dipole=dipole)

    return rhs
