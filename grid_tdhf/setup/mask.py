from grid_tdhf.utils import select_keys
from grid_tdhf.masks import MASK_REGISTRY


def setup_mask(simulation_config):
    mask_name = simulation_config.mask_name

    if mask_name == "no-mask":
        return None

    params = {**vars(simulation_config)}

    if mask_name not in MASK_REGISTRY:
        raise ValueError(f"Mask {mask_name} is not available.")

    entry = MASK_REGISTRY[mask_name]
    mask_func = entry["func"]
    required_params = entry["required_params"]

    missing = required_params - params.keys()
    if missing:
        raise ValueError(f"Missing required mask parameters: {missing}")

    mask_args = select_keys(params, required_params)

    return mask_func(**mask_args)
