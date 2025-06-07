from grid_tdhf.properties import (
    compute_expec_z,
    compute_norm,
    compute_energy,
)


class PropertiesComputer:
    def __init__(self, simulation_config, potential_computer):
        self.config = simulation_config
        self.potential_computer = potential_computer

    def compute_expec_z(self, state):
        config = self.config

        return compute_expec_z(
            state, config.z_Omega, config.r, config.weights, config.m_list
        )

    def compute_norm(self, state):
        return compute_norm(state, self.config.weights)

    def compute_energy(self, state):
        config = self.config

        return compute_energy(
            state,
            self.potential_computer.V_d_electron,
            self.potential_computer.V_d_positron,
            self.potential_computer.V_x,
            n_orbs=config.n_orbs,
            m_list=config.m_list,
            has_positron=config.has_positron,
            coulomb_potential=config.coulomb_potential,
            centrifugal_potential_l=config.centrifugal_potential_l,
            centrifugal_potential_r=config.centrifugal_potential_r,
            D2=config.D2,
            weights=config.weights,
        )
