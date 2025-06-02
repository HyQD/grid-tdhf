from grid_tdhf.utils import select_keys


def setup_rhs(
    inputs,
    system_info,
    aux_arrays,
    rme,
    angular_matrices,
    laser_obj,
    potential_computer,
):

    args = {
        **dict(vars(inputs)),
        **dict(vars(system_info)),
        **dict(vars(aux_arrays)),
        **dict(vars(rme)),
        **dict(vars(angular_matrices)),
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
    dipole = RHSDipole(
        **select_keys(args, RHSDipole.required_params), laser_obj=laser_obj
    )

    rhs = CompositeRHS(core=core, mean_field=mean_field, dipole=dipole)

    return rhs
