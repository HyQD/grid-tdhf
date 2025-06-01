def select_keys(d, keys):
    """Return a subset of dictionary `d` with only the specified `keys`."""
    return {k: d[k] for k in keys if k in d}
