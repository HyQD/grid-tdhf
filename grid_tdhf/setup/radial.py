from types import SimpleNamespace

from grid_methods.pseudospectral_grids.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map,
)

from grid_methods.spherical_coordinates.radial_matrix_elements import (
    RadialMatrixElements,
)


def setup_radial_arrays(inputs):
    N = inputs.N
    r_max = inputs.r_max

    gauss_legendre_lobatto = GaussLegendreLobatto(N, Linear_map(r_max=r_max))
    radial_matrix_elements = RadialMatrixElements(gauss_legendre_lobatto)

    radial_arrays = {**vars(gauss_legendre_lobatto), **vars(radial_matrix_elements)}

    return SimpleNamespace(**radial_arrays)
