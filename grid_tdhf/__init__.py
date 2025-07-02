import grid_lib

REQUIRED_GRID_METHODS_VERSION = "1.0.0"

if getattr(grid_lib, "__version__", None) != REQUIRED_GRID_METHODS_VERSION:
    raise ImportError(
        f"grid_lib version {REQUIRED_GRID_METHODS_VERSION} required, "
        f"but {getattr(grid_lib, '__version__', 'unknown')} is installed."
    )
