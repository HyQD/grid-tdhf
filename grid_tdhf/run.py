import numpy as np

from grid_tdhf.inputs import parse_arguments
from grid_tdhf.setup.system_info import get_atomic_system_params
from grid_tdhf.setup.angular import setup_angular_matrices
from grid_tdhf.setup.radial import setup_radial_matrices
from grid_tdhf.setup.auxiliary_arrays import setup_auxiliary_arrays
from grid_tdhf.setup.laser import setup_laser
from grid_tdhf.setup.init_state import setup_init_state
from grid_tdhf.setup.rhs import setup_rhs
from grid_tdhf.potentials import PotentialComputer


def main():
    inputs = parse_arguments(verbose=False)

    system_info = get_atomic_system_params(inputs)

    angular_matrices = setup_angular_matrices(inputs, system_info)

    gll, rme = setup_radial_matrices(inputs)

    aux_arrays = setup_auxiliary_arrays(inputs, system_info, gll, rme)

    laser_obj = setup_laser(inputs)

    potential_computer = PotentialComputer(inputs, system_info, aux_arrays, rme)

    u = setup_init_state(inputs, system_info, aux_arrays, gll, rme)

    potential_computer.set_state(u)
    potential_computer.construct_potentials(u)

    rhs = setup_rhs(
        inputs,
        system_info,
        aux_arrays,
        rme,
        angular_matrices,
        laser_obj,
        potential_computer,
    )

    rhs(u, 0)


if __name__ == "__main__":
    main()
