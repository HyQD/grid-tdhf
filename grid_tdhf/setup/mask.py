from grid_tdhf.utils import resolve_required_params
from grid_tdhf.masks import MASK_REGISTRY


def setup_mask(simulation_config, used_inputs=None, param_mapping=None):
    if param_mapping is None:
        param_mapping = {
            "margin": "mask_margin",
            "n": "mask_n",
        }

    mask_name = simulation_config.mask_name

    if mask_name == "no-mask":
        return None

    if mask_name not in MASK_REGISTRY:
        raise ValueError(f"Mask {mask_name} is not available.")

    params = {**vars(simulation_config)}

    entry = MASK_REGISTRY[mask_name]
    mask_func = entry["func"]
    required_params = entry["required_params"]

    mask_args = resolve_required_params(
        required_params, params, used_inputs, param_mapping
    )

    missing = required_params - mask_args.keys()
    if missing:
        raise ValueError(f"Missing required mask parameters: {missing}")

    return mask_func(**mask_args)
