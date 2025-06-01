from grid_methods.spherical_coordinates.gauss_legendre_lobatto import (
    GaussLegendreLobatto,
    Linear_map,
)

from grid_methods.spherical_coordinates.radial_matrix_elements import (
    RadialMatrixElements,
)


def setup_radial_matrices(inputs):
    N = inputs.N
    r_max = inputs.r_max

    gll = GaussLegendreLobatto(N, Linear_map(r_max=r_max))
    rme = RadialMatrixElements(gll)

    gll, rme

    return gll, rme
