import numpy as np
from opt_einsum import contract

from grid_tdhf.potentials import (
    compute_direct_potential,
    compute_exchange_potential_for_orbital,
)


class PotentialComputer:
    def __init__(self, comm, simulation_config):
        rank = comm.Get_rank()
        self.rank = rank

        self.n_orbs_tot = simulation_config.n_orbs_tot
        self.nl = simulation_config.nl
        self.nr = simulation_config.nr

        self.m_list = simulation_config.m_list
        self.m_list_tot = simulation_config.m_list_tot
        self.m_max = simulation_config.m_max
        self.has_positron = simulation_config.has_positron

        self.poisson_inverse = simulation_config.poisson_inverse
        self.gaunt_dict = simulation_config.gaunt_dict

        self.u = None
        self.V_d_electron = None
        self.V_d_positron = None
        self.V_x = None

    def set_state(self, u):
        self.u = u.copy()

    def compute_direct_potential(self):
        self.V_d_electron, self.V_d_positron = compute_direct_potential(
            self.u,
            self.n_orbs_tot,
            self.nl,
            self.nr,
            self.m_list_tot,
            self.m_max,
            self.poisson_inverse,
            self.gaunt_dict,
            self.has_positron,
        )

    def compute_exchange_potential(self, u):
        self.V_x = compute_exchange_potential_for_orbital(
            self.u,
            u[0],
            self.n_orbs_tot,
            self.nl,
            self.nr,
            self.m_list_tot,
            self.m_list[0],
            self.poisson_inverse,
            self.gaunt_dict,
        )
