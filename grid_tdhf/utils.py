def resolve_required_params(
    required_keys, params, used_inputs=None, param_mapping=None, validate=False
):
    used_inputs = used_inputs or set()
    param_mapping = param_mapping or {}
    result = {}

    for key in required_keys:
        actual_key = param_mapping.get(key, key)
        if actual_key in params:
            result[key] = params[actual_key]
            used_inputs.add(actual_key)
        else:
            if validate:
                raise KeyError(
                    f"Required parameter '{actual_key}' for key '{key}' is missing."
                )
    return result


def select_keys(d, keys, exclude={}):
    return {k: d[k] for k in keys if k in d and k not in exclude}
