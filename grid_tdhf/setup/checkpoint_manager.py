from grid_tdhf.io.checkpoint import CheckpointManager


def setup_checkpoint_manager(fileroot, sampler, inputs, simulation_info):
    return CheckpointManager(
        fileroot,
        sampler,
        inputs,
        inputs.checkpoint_interval,
        simulation_info.total_steps,
    )
