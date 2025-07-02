import numpy as np
from opt_einsum import contract

from grid_lib.spherical_coordinates.Hpsi_components import pz_psi


class CompositeRHS:
    def __init__(self, **rhs_components):
        self.components = rhs_components

        for name, component in rhs_components.items():
            setattr(self, name, component)

    def __call__(self, u, t, ravel=True):
        return sum(
            (component(u, t, ravel=ravel) for component in self.components.values())
        )


class RHSCore:
    required_params = {
        "nl",
        "nr",
        "D2",
        "coulomb_potential",
        "centrifugal_potential_l",
        "centrifugal_potential_r",
        "has_positron",
        "is_positron",
    }

    def __init__(
        self,
        nl,
        nr,
        D2,
        coulomb_potential,
        centrifugal_potential_l,
        centrifugal_potential_r,
        has_positron=False,
        is_positron=False,
    ):
        self.T_D2 = -(1 / 2) * D2
        self.coulomb_potential = coulomb_potential
        self.centrifugal_potential_l = centrifugal_potential_l
        self.centrifugal_potential_r = centrifugal_potential_r
        self.nl = nl
        self.nr = nr
        self.has_positron = has_positron
        self.is_positron = is_positron

    def __call__(self, u, t, ravel=True):
        u = u.reshape(1, self.nl, self.nr)
        u_new = np.zeros((1, self.nl, self.nr), dtype=np.complex128)

        if not self.is_positron:
            u_new[0] += contract("Ij, ij->Ii", u[0], self.T_D2)
            u_new[0] += contract("Ik, k->Ik", u[0], self.coulomb_potential)
            u_temp = contract("I, Ii->Ii", self.centrifugal_potential_l, u[0])
            u_new[0] += contract("i, Ii->Ii", self.centrifugal_potential_r, u_temp)

        else:
            u_new[0] += contract("Ij, ij->Ii", u[0], self.T_D2)
            u_new[0] -= contract("Ik, k->Ik", u[0], self.coulomb_potential)
            u_temp = contract("I, Ii->Ii", self.centrifugal_potential_l, u[0])
            u_new[0] += contract("i, Ii->Ii", self.centrifugal_potential_r, u_temp)

        return u_new.ravel() if ravel else u_new


class RHSMeanField:
    required_params = {
        "potential_computer",
        "n_orbs_tot",
        "nl",
        "nr",
        "m_list",
        "has_positron",
        "is_positron",
    }

    def __init__(
        self,
        potential_computer,
        n_orbs_tot,
        nl,
        nr,
        m_list,
        has_positron=False,
        is_positron=False,
    ):
        self.potential_computer = potential_computer
        self.n_orbs_tot = n_orbs_tot
        self.nl = nl
        self.nr = nr
        self.m_list = m_list
        self.has_positron = has_positron
        self.is_positron = is_positron
        self.single_orbital = False if n_orbs_tot > 1 else True

    def __call__(self, u, t, ravel=True):
        u = u.reshape(1, self.nl, self.nr)
        u_new = np.zeros((1, self.nl, self.nr), dtype=np.complex128)

        if not self.single_orbital or self.is_positron:
            self.potential_computer.compute_exchange_potential(u)

        V_d_electron = self.potential_computer.V_d_electron
        V_d_positron = self.potential_computer.V_d_positron
        V_x = self.potential_computer.V_x
        u_bar = self.potential_computer.u

        if not self.is_positron:
            m = self.m_list[0]

            if self.single_orbital:
                u_new[0] += contract("ijr,jr->ir", V_d_electron[m], u[0])
            else:
                u_new[0] += 2 * contract("ijr,jr->ir", V_d_electron[m], u[0])

                for j in range(self.n_orbs_tot):
                    u_new[0] -= contract("ijr,jr->ir", V_x[j], u_bar[j])

            if self.has_positron:
                m = self.m_list[0]
                u_new[0] -= contract("ijr,jr->ir", V_d_positron[m], u[0])

        else:
            u_new[0] -= 2 * contract("ijr,jr->ir", V_d_electron[0], u[0])

        return u_new.ravel() if ravel else u_new


class RHSDipoleVelocityGauge:
    required_params = {
        "n_orbs",
        "nl",
        "nr",
        "laser",
        "z_Omega",
        "H_z_beta",
        "r",
        "D1",
        "m_list",
        "has_positron",
        "is_positron",
    }

    def __init__(
        self,
        n_orbs,
        nl,
        nr,
        laser,
        z_Omega,
        H_z_beta,
        r,
        D1,
        m_list,
        has_positron=False,
        is_positron=False,
    ):
        self.n_orbs = n_orbs
        self.nl = nl
        self.nr = nr
        self.laser = laser
        self.z_Omega = z_Omega
        self.H_z_beta = H_z_beta
        self.D1 = D1
        self.r_inv = 1 / r
        self.m_list = m_list
        self.has_positron = has_positron
        self.is_positron = is_positron
        self.N_orbs = n_orbs + 1 if has_positron else n_orbs

    def __call__(self, u, t, ravel=True):
        u = u.reshape(1, self.nl, self.nr)
        u_new = np.zeros((1, self.nl, self.nr), dtype=np.complex128)

        if not self.is_positron:
            m = self.m_list[0]

            du_dr = contract("ij, Ij->Ii", self.D1, u[0])

            u_new[0] = self.laser(t) * pz_psi(
                u[0], du_dr, self.z_Omega[m], self.H_z_beta[m], self.r_inv
            )

        else:
            du_dr = contract("ij, Ij->Ii", self.D1, u[0])

            u_new[0] = -self.laser(t) * pz_psi(
                u[0], du_dr, self.z_Omega[0], self.H_z_beta[0], self.r_inv
            )

        return u_new.ravel() if ravel else u_new


class RHSDipoleLengthGauge:
    required_params = {
        "n_orbs",
        "nl",
        "nr",
        "laser",
        "z_Omega",
        "r",
        "m_list",
        "has_positron",
        "is_positron",
    }

    def __init__(
        self,
        n_orbs,
        nl,
        nr,
        laser,
        z_Omega,
        r,
        m_list,
        has_positron=False,
        is_positron=False,
    ):
        self.n_orbs = n_orbs
        self.nl = nl
        self.nr = nr
        self.laser = laser
        self.z_Omega = z_Omega
        self.r = r
        self.m_list = m_list
        self.has_positron = has_positron
        self.is_positron = is_positron
        self.N_orbs = n_orbs + 1 if has_positron else n_orbs

    def __call__(self, u, t, ravel=True):
        u = u.reshape(self.N_orbs, self.nl, self.nr)
        u_new = np.zeros((self.N_orbs, self.nl, self.nr), dtype=np.complex128)

        if not self.is_positron:
            m = self.m_list[0]

            u_temp = contract("IJ, Jk->Ik", self.z_Omega[m], u[0])
            u_new[0] += self.laser(t) * contract("Ik, k->Ik", u_temp, self.r)

        else:
            u_temp = contract("IJ, Jk->Ik", self.z_Omega[0], u[0])
            u_new[0] -= self.laser(t) * contract("Ik, k->Ik", u_temp, self.r)

        return u_new.ravel() if ravel else u_new
