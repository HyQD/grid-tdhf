import numpy as np
from opt_einsum import contract

from grid_tdhf.potentials import (
    compute_direct_potential,
    compute_exchange_potential,
)


class PotentialComputer:
    def __init__(self, inputs, system_info, radial_arrays, aux_arrays):
        self.n_orbs = system_info.n_orbs
        self.nl = inputs.nl
        self.nr = radial_arrays.nr

        self.m_list = system_info.m_list
        self.m_max = system_info.m_max
        self.has_positron = system_info.has_positron

        self.poisson_inverse = aux_arrays.poisson_inverse
        self.gaunt_dict = aux_arrays.gaunt_dict

        self.u = None
        self.V_d_electron = None
        self.V_d_positron = None
        self.V_x = None

    def set_state(self, u):
        self.u = u.copy()

    def compute_direct_potential(self):
        self.V_d_electron, self.V_d_positron = compute_direct_potential(
            self.u,
            self.n_orbs,
            self.nl,
            self.nr,
            self.m_list,
            self.m_max,
            self.poisson_inverse,
            self.gaunt_dict,
            self.has_positron,
        )

    def compute_exchange_potential(self, u):
        self.V_x = compute_exchange_potential(
            self.u,
            u,
            self.n_orbs,
            self.nl,
            self.nr,
            self.m_list,
            self.poisson_inverse,
            self.gaunt_dict,
        )
