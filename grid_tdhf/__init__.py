import grid_lib

from packaging import version

MINIMUM_GRID_LIB_VERSION = "1.1.0"

_installed_grid_lib_version = getattr(grid_lib, "__version__", None)

if _installed_grid_lib_version is None or version.parse(
    _installed_grid_lib_version
) < version.parse(MINIMUM_GRID_LIB_VERSION):
    raise ImportError(
        f"grid_lib >= {MINIMUM_GRID_LIB_VERSION} required, "
        f"but {_installed_grid_lib_version or 'unknown'} is installed."
    )
