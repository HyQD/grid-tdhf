from grid_tdhf.io.checkpoint import CheckpointManager


def setup_checkpoint_manager(
    fileroot, sampler, inputs, simulation_config, simulation_info
):
    return CheckpointManager(
        fileroot,
        sampler,
        inputs,
        simulation_config.full_state,
        simulation_config.active_orbitals,
        inputs.checkpoint_interval,
        simulation_info.total_steps,
    )
