from grid_tdhf.properties import (
    compute_expec_z,
    compute_norm,
    compute_energy,
)


class PropertiesComputer:
    def __init__(self, simulation_config, potential_computer):
        self.n_orbs = simulation_config.n_orbs
        self.m_list = simulation_config.m_list
        self.has_positron = simulation_config.has_positron

        self.z_Omega = simulation_config.z_Omega

        self.r = simulation_config.r
        self.D2 = simulation_config.D2
        self.weights = simulation_config.weights

        self.coulomb_potential = simulation_config.coulomb_potential
        self.centrifugal_potential_l = simulation_config.centrifugal_potential_l
        self.centrifugal_potential_r = simulation_config.centrifugal_potential_r

        self.potential_computer = potential_computer

    def compute_expec_z(self, state):
        return compute_expec_z(state, self.z_Omega, self.r, self.weights, self.m_list)

    def compute_norm(self, state):
        return compute_norm(state, self.weights)

    def compute_energy(self, state):
        return compute_energy(
            state,
            self.potential_computer.V_d_electron,
            self.potential_computer.V_d_positron,
            self.potential_computer.V_x,
            n_orbs=self.n_orbs,
            m_list=self.m_list,
            has_positron=self.has_positron,
            coulomb_potential=self.coulomb_potential,
            centrifugal_potential_l=self.centrifugal_potential_l,
            centrifugal_potential_r=self.centrifugal_potential_r,
            D2=self.D2,
            weights=self.weights,
        )
