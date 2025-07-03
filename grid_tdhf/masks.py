import numpy as np


def cosine_mask(*, r, r_max, margin, n):
    mask_r = np.zeros(len(r))

    r0 = r_max - margin

    ind1 = r < r0
    ind2 = r == r_max
    ind3 = np.invert(ind1 + ind2)

    mask_r[ind1] = 1
    mask_r[ind2] = 0
    mask_r[ind3] = np.cos(np.pi * (r[ind3] - r0) / (2 * (r_max - r0))) ** (1 / n)

    return mask_r


MASK_REGISTRY = {
    "cosine_mask": {
        "func": cosine_mask,
        "required_params": {"r", "r_max", "margin", "n"},
    },
}
