import numpy as np
from types import SimpleNamespace

from grid_methods.spherical_coordinates.angular_matrix_elements import (
    AngularMatrixElements_l,
)


def setup_angular_matrices(inputs, system_info, array_names=["z_Omega", "H_z_beta"]):
    l_max = inputs.l_max
    m_max = system_info.m_max

    angular_matrices = {}

    for name in array_names:
        angular_matrices[name] = {}

    m_values = np.arange(-m_max, m_max + 1)

    for m in m_values:
        angular_matrices_m = AngularMatrixElements_l(
            arr_to_calc=array_names, l_max=l_max, m=m
        )

        for name in array_names:
            angular_matrices[name][m] = angular_matrices_m(name)

    return SimpleNamespace(**angular_matrices)
