def select_keys(d, keys, exclude={}):
    """Return a subset of dictionary `d` with only the specified `keys`,
    excluding any in `exclude`."""
    return {k: d[k] for k in keys if k in d and k not in exclude}
