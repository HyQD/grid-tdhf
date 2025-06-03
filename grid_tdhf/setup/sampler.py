from grid_tdhf.io.sampling import Sampler

from grid_tdhf.utils import select_keys


def setup_sampler(properties_computer, inputs):

    params = {**vars(inputs)}

    sampler_args = select_keys(params, Sampler.required_params)

    return Sampler(**sampler_args, properties_computer=properties_computer)
