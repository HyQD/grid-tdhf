from grid_tdhf.io.checkpoint import CheckpointManager


def setup_checkpoint_manager(sampler, inputs, simulation_config, simulation_info):
    fileroot = simulation_config.fileroot
    output_dir = simulation_config.output_dir

    return CheckpointManager(
        fileroot,
        sampler,
        inputs,
        simulation_config.full_state,
        simulation_config.active_orbitals,
        inputs.checkpoint_interval,
        simulation_info.total_steps,
        output_dir=output_dir,
    )
