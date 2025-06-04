import uuid

from grid_tdhf.inputs import parse_arguments
from grid_tdhf.setup.system_info import get_atomic_system_params
from grid_tdhf.setup.angular import setup_angular_arrays
from grid_tdhf.setup.radial import setup_radial_arrays
from grid_tdhf.setup.auxiliary_arrays import setup_auxiliary_arrays
from grid_tdhf.setup.laser import setup_laser
from grid_tdhf.setup.init_state import setup_init_state
from grid_tdhf.setup.rhs import setup_rhs
from grid_tdhf.computers.potential_computer import PotentialComputer
from grid_tdhf.computers.properties_computer import PropertiesComputer
from grid_tdhf.setup.integrator import setup_integrator
from grid_tdhf.setup.sampler import setup_sampler
from grid_tdhf.setup.simulation import setup_simulation, run_simulation
from grid_tdhf.setup.mask import setup_mask
from grid_tdhf.setup.checkpoint_manager import setup_checkpoint_manager
from grid_tdhf.setup.simulation_config import generate_simulation_config
from grid_tdhf.setup.freeze_config import generate_freeze_config


def main():
    inputs = parse_arguments(verbose=False)

    fileroot = str(uuid.uuid4())

    system_info = get_atomic_system_params(inputs)

    angular_arrays = setup_angular_arrays(inputs, system_info)
    radial_arrays = setup_radial_arrays(inputs)
    aux_arrays = setup_auxiliary_arrays(inputs, system_info, radial_arrays)

    u = setup_init_state(inputs, system_info, angular_arrays, radial_arrays, aux_arrays)

    if inputs.gs_only:
        return

    config_overrides = generate_freeze_config(u, inputs, system_info, aux_arrays)
    simulation_config = generate_simulation_config(
        u,
        inputs,
        system_info,
        angular_arrays,
        radial_arrays,
        aux_arrays,
        overrides=config_overrides,
    )

    integrator = setup_integrator(simulation_config)

    potential_computer = PotentialComputer(simulation_config)
    properties_computer = PropertiesComputer(simulation_config, potential_computer)

    laser = setup_laser(simulation_config)
    rhs = setup_rhs(simulation_config, potential_computer, laser)

    mask = setup_mask(simulation_config)

    simulation_info = setup_simulation(simulation_config)

    sampler = setup_sampler(simulation_config, properties_computer)
    checkpoint_manager = setup_checkpoint_manager(
        fileroot, sampler, inputs, simulation_info
    )

    run_simulation(
        integrator,
        rhs,
        mask,
        potential_computer,
        sampler,
        checkpoint_manager,
        simulation_config,
        simulation_info,
    )


if __name__ == "__main__":
    main()
