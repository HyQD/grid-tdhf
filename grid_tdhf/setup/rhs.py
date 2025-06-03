from grid_tdhf.utils import select_keys


def setup_rhs(
    inputs,
    system_info,
    angular_matrices,
    radial_arrays,
    aux_arrays,
    potential_computer,
    laser=None,
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

    core = RHSCore(**select_keys(args, RHSCore.required_params))
    mean_field = RHSMeanField(
        **select_keys(args, RHSMeanField.required_params),
        potential_computer=potential_computer
    )

    kwargs = {"core": core, "mean_field": mean_field}

    if laser is not None:
        if inputs.gauge == "velocity":
            from grid_tdhf.rhs import RHSDipoleVelocityGauge as RHSDipole
        elif inputs.gauge == "length":
            from grid_tdhf.rhs import RHSDipoleLengthGauge as RHSDipole

        dipole = RHSDipole(**select_keys(args, RHSDipole.required_params), laser=laser)
        kwargs["dipole"] = dipole

    rhs = CompositeRHS(**kwargs)

    return rhs
