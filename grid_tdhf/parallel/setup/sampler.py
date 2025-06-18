from grid_tdhf.parallel.io.sampling import Sampler

from grid_tdhf.utils import select_keys


def setup_sampler(simulation_config, properties_computer):

    params = {**vars(simulation_config)}

    sampler_args = select_keys(params, Sampler.required_params)

    return Sampler(**sampler_args, properties_computer=properties_computer)
