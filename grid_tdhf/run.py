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
from grid_tdhf.setup.load_run import resume_from_checkpoint
from grid_tdhf.config.system import generate_system_config, generate_runtime_config
from grid_tdhf.config.simulation import get_simulation_overrides
from grid_tdhf.config.freeze import get_freeze_overrides


def main():
    inputs = parse_arguments(verbose=False)
    used_inputs = set()

    fileroot = inputs.load_run if inputs.load_run is not None else str(uuid.uuid4())

    system_info = get_atomic_system_params(inputs)

    angular_arrays = setup_angular_arrays(inputs, system_info)
    radial_arrays = setup_radial_arrays(inputs)
    aux_arrays = setup_auxiliary_arrays(inputs, system_info, radial_arrays)

    system_config = generate_system_config(
        inputs, system_info, angular_arrays, radial_arrays, aux_arrays
    )

    u = setup_init_state(system_config, used_inputs)

    if inputs.gs_only:
        return

    simulation_overrides = get_simulation_overrides(u, system_config)
    freeze_overrides = get_freeze_overrides(u, system_config)
    overrides = {**simulation_overrides, **freeze_overrides}
    simulation_config = generate_runtime_config(system_config, overrides)

    potential_computer = PotentialComputer(simulation_config)
    properties_computer = PropertiesComputer(simulation_config, potential_computer)

    integrator = setup_integrator(
        simulation_config, potential_computer, used_inputs=used_inputs
    )

    laser = setup_laser(simulation_config, used_inputs=used_inputs)
    rhs = setup_rhs(
        simulation_config, potential_computer, laser, used_inputs=used_inputs
    )

    mask = setup_mask(simulation_config, used_inputs=used_inputs)

    simulation_info = setup_simulation(simulation_config)

    sampler = setup_sampler(simulation_config, properties_computer)
    checkpoint_manager = setup_checkpoint_manager(
        fileroot, sampler, inputs, simulation_config, simulation_info
    )

    if inputs.load_run:
        simulation_info, sampler = resume_from_checkpoint(
            fileroot, simulation_info, sampler
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
