import grid_methods

REQUIRED_GRID_METHODS_VERSION = "1.0.0"

if getattr(grid_methods, "__version__", None) != REQUIRED_GRID_METHODS_VERSION:
    raise ImportError(
        f"grid_methods version {REQUIRED_GRID_METHODS_VERSION} required, "
        f"but {getattr(grid_methods, '__version__', 'unknown')} is installed."
    )
