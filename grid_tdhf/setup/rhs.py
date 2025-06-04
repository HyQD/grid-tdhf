from grid_tdhf.utils import select_keys

from grid_tdhf.rhs import CompositeRHS
from grid_tdhf.rhs import RHSCore
from grid_tdhf.rhs import RHSMeanField


def setup_rhs(
    simulation_config,
    potential_computer,
    laser=None,
):

    params = {**vars(simulation_config)}

    core = RHSCore(**select_keys(params, RHSCore.required_params))
    mean_field = RHSMeanField(
        **select_keys(params, RHSMeanField.required_params),
        potential_computer=potential_computer
    )

    kwargs = {"core": core, "mean_field": mean_field}

    if laser is not None:
        if simulation_config.gauge == "velocity":
            from grid_tdhf.rhs import RHSDipoleVelocityGauge as RHSDipole
        elif simulation_config.gauge == "length":
            from grid_tdhf.rhs import RHSDipoleLengthGauge as RHSDipole

        dipole = RHSDipole(
            **select_keys(params, RHSDipole.required_params), laser=laser
        )
        kwargs["dipole"] = dipole

    rhs = CompositeRHS(**kwargs)

    return rhs
