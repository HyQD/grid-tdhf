import numpy as np

from grid_tdhf.inputs import parse_arguments
from grid_tdhf.setup.system_info import get_atomic_system_params
from grid_tdhf.setup.angular import setup_angular_arrays
from grid_tdhf.setup.radial import setup_radial_arrays
from grid_tdhf.setup.auxiliary_arrays import setup_auxiliary_arrays
from grid_tdhf.setup.laser import setup_laser
from grid_tdhf.setup.init_state import setup_init_state
from grid_tdhf.setup.rhs import setup_rhs
from grid_tdhf.potentials import PotentialComputer
from grid_tdhf.setup.integrator import setup_integrator


def main():
    inputs = parse_arguments(verbose=False)

    system_info = get_atomic_system_params(inputs)

    angular_arrays = setup_angular_arrays(inputs, system_info)
    radial_arrays = setup_radial_arrays(inputs)
    aux_arrays = setup_auxiliary_arrays(inputs, system_info, radial_arrays)

    laser_obj = setup_laser(inputs)

    potential_computer = PotentialComputer(
        inputs, system_info, radial_arrays, aux_arrays
    )

    u = setup_init_state(inputs, system_info, radial_arrays, aux_arrays)

    potential_computer.set_state(u)
    potential_computer.construct_potentials(u)

    integrator = setup_integrator(inputs, system_info, radial_arrays)

    rhs = setup_rhs(
        inputs,
        system_info,
        angular_arrays,
        radial_arrays,
        aux_arrays,
        laser_obj,
        potential_computer,
    )


if __name__ == "__main__":
    main()
