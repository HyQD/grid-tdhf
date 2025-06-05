import numpy as np


def resume_from_checkpoint(fileroot, simulation_info, sampler):
    _, metadata = load_info(fileroot)

    simulation_info.init_step = metadata["current_step"]
    simulation_info.t0 = metadata["current_time"]

    loaded_samples = load_samples(fileroot)
    sampler.samples.update(loaded_samples)

    return simulation_info, sampler


def load_info(fileroot, direc="output/", tag="_info"):
    info = np.load(direc + fileroot + tag + ".npz", allow_pickle=True)

    inputs = info["inputs"].item()
    metadata = info["metadata"].item()

    return inputs, metadata


def load_samples(fileroot, direc="output/", tag="_samples"):
    loaded = np.load(direc + fileroot + tag + ".npz", allow_pickle=True)
    samples = {}

    for key in loaded.files:
        array = loaded[key]
        samples[key] = [array[i].copy() for i in range(len(array))]

    return samples


def load_state(fileroot, direc="output/", tag="_state"):
    states = np.load(direc + fileroot + tag + ".npz", allow_pickle=True)

    try:
        state = states["state"]
    except:
        state = None

    try:
        state_time_points = states["state_time_points"]
    except:
        state_time_points = None

    try:
        u = states["u"]
    except:
        u = None

    return u, state, state_time_points
