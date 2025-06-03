from grid_tdhf.properties import (
    compute_expec_z,
    compute_norm,
    compute_energy,
)


class PropertiesComputer:
    def __init__(
        self, system_info, angular_arrays, radial_arrays, aux_arrays, potential_computer
    ):
        self.n_orbs = system_info.n_orbs
        self.m_list = system_info.m_list
        self.has_positron = system_info.has_positron

        self.z_Omega = angular_arrays.z_Omega

        self.r = radial_arrays.r
        self.D2 = radial_arrays.D2
        self.weights = radial_arrays.weights

        self.coulomb_potential = aux_arrays.coulomb_potential
        self.centrifugal_potential_l = aux_arrays.centrifugal_potential_l
        self.centrifugal_potential_r = aux_arrays.centrifugal_potential_r

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
            self.n_orbs,
            self.m_list,
            self.has_positron,
            self.coulomb_potential,
            self.centrifugal_potential_l,
            self.centrifugal_potential_r,
            self.D2,
            self.weights,
        )
