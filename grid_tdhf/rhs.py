import numpy as np
from opt_einsum import contract

from grid_methods.spherical_coordinates.Hpsi_components import pz_psi


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
        "n_orbs",
        "nl",
        "nr",
        "D2",
        "coulomb_potential",
        "centrifugal_potential_l",
        "centrifugal_potential_r",
        "n_frozen_orbitals",
        "has_positron",
    }

    def __init__(
        self,
        n_orbs,
        nl,
        nr,
        D2,
        coulomb_potential,
        centrifugal_potential_l,
        centrifugal_potential_r,
        n_frozen_orbitals,
        has_positron=False,
    ):
        self.T_D2 = -(1 / 2) * D2
        self.coulomb_potential = coulomb_potential
        self.centrifugal_potential_l = centrifugal_potential_l
        self.centrifugal_potential_r = centrifugal_potential_r
        self.n_orbs = n_orbs
        self.nl = nl
        self.nr = nr
        self.n_frozen_orbitals = n_frozen_orbitals
        self.has_positron = has_positron
        self.N_orbs = n_orbs + 1 if has_positron else n_orbs

    def __call__(self, u, t, ravel=True):
        u = u.reshape(self.N_orbs, self.nl, self.nr)
        u_new = np.zeros((self.N_orbs, self.nl, self.nr), dtype=np.complex128)

        for p in range(self.n_frozen_orbitals, self.n_orbs):
            u_new[p] += contract("Ij, ij->Ii", u[p], self.T_D2)
            u_new[p] += contract("Ik, k->Ik", u[p], self.coulomb_potential)
            u_temp = contract("I, Ii->Ii", self.centrifugal_potential_l, u[p])
            u_new[p] += contract("i, Ii->Ii", self.centrifugal_potential_r, u_temp)

        if self.has_positron:
            u_new[-1] += contract("Ij, ij->Ii", u[-1], self.T_D2)
            u_new[-1] -= contract("Ik, k->Ik", u[-1], self.coulomb_potential)
            u_temp = contract("I, Ii->Ii", self.centrifugal_potential_l, u[-1])
            u_new[-1] += contract("i, Ii->Ii", self.centrifugal_potential_r, u_temp)

        return u_new.ravel() if ravel else u_new


class RHSMeanField:
    required_params = {
        "potential_computer",
        "n_orbs",
        "nl",
        "nr",
        "n_frozen_orbitals",
        "nL",
        "m_list",
        "has_positron",
    }

    def __init__(
        self,
        potential_computer,
        n_orbs,
        nl,
        nr,
        n_frozen_orbitals,
        nL,
        m_list,
        has_positron=False,
    ):
        self.potential_computer = potential_computer
        self.n_orbs = n_orbs
        self.nl = nl
        self.nr = nr
        self.m_list = m_list
        self.n_frozen_orbitals = n_frozen_orbitals
        self.nL = nL
        self.has_positron = has_positron
        self.N_orbs = n_orbs + 1 if has_positron else n_orbs
        self.single_orbital = False if n_orbs > 1 else True

    def __call__(self, u, t, ravel=True):
        u = u.reshape(self.N_orbs, self.nl, self.nr)
        u_new = np.zeros((self.N_orbs, self.nl, self.nr), dtype=np.complex128)

        if not self.single_orbital:
            self.potential_computer.compute_exchange_potential(u)

        V_d_electron = self.potential_computer.V_d_electron
        V_d_positron = self.potential_computer.V_d_positron
        V_x = self.potential_computer.V_x
        u_bar = self.potential_computer.u

        for p in range(self.n_frozen_orbitals, self.n_orbs):
            m = self.m_list[p]

            if self.single_orbital:
                u_new[p] += contract("ijr,jr->ir", V_d_electron[m], u[p])
            else:
                u_new[p] += 2 * contract("ijr,jr->ir", V_d_electron[m], u[p])

                for j in range(self.n_orbs):
                    u_new[p] -= contract("ijr,jr->ir", V_x[j, p], u_bar[j])

        if self.has_positron:
            for p in range(self.n_frozen_orbitals, self.n_orbs):
                m = self.m_list[p]

                u_new[p] -= contract("ijr,jr->ir", V_d_positron[m], u[p])

            u_new[-1] -= 2 * contract("ijr,jr->ir", V_d_electron["0"], u[-1])

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
        "n_frozen_orbitals",
        "m_list",
        "has_positron",
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
        n_frozen_orbitals,
        m_list,
        has_positron=False,
    ):
        self.n_orbs = n_orbs
        self.nl = nl
        self.nr = nr
        self.laser = laser
        self.z_Omega = z_Omega
        self.H_z_beta = H_z_beta
        self.D1 = D1
        self.r_inv = 1 / r
        self.n_frozen_orbitals = n_frozen_orbitals
        self.m_list = m_list
        self.has_positron = has_positron
        self.N_orbs = n_orbs + 1 if has_positron else n_orbs

    def __call__(self, u, t, ravel=True):
        u = u.reshape(self.N_orbs, self.nl, self.nr)
        u_new = np.zeros((self.N_orbs, self.nl, self.nr), dtype=np.complex128)

        for p in range(self.n_frozen_orbitals, self.n_orbs):
            m = self.m_list[p]

            du_dr = contract("ij, Ij->Ii", self.D1, u[p])

            u_new[p] = self.laser(t) * pz_psi(
                u[p], du_dr, self.z_Omega[m], self.H_z_beta[m], self.r_inv
            )

        if self.has_positron:
            du_dr = contract("ij, Ij->Ii", self.D1, u[-1])

            u_new[-1] = -self.laser(t) * pz_psi(
                u[-1], du_dr, self.z_Omega["0"], self.H_z_beta["0"], self.r_inv
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
        "n_frozen_orbitals",
        "m_list",
        "has_positron",
    }

    def __init__(
        self,
        n_orbs,
        nl,
        nr,
        laser,
        z_Omega,
        r,
        n_frozen_orbitals,
        m_list,
        has_positron=False,
    ):
        self.n_orbs = n_orbs
        self.nl = nl
        self.nr = nr
        self.laser = laser
        self.z_Omega = z_Omega
        self.r = r
        self.n_frozen_orbitals = n_frozen_orbitals
        self.m_list = m_list
        self.has_positron = has_positron
        self.N_orbs = n_orbs + 1 if has_positron else n_orbs

    def __call__(self, u, t, ravel=True):
        u = u.reshape(self.N_orbs, self.nl, self.nr)
        u_new = np.zeros((self.N_orbs, self.nl, self.nr), dtype=np.complex128)

        for p in range(self.n_frozen_orbitals, self.n_orbs):
            m = self.m_list[p]

            u_temp = contract("IJ, Jk->Ik", self.z_Omega[m], u[p])
            u_new[p] += self.laser(t) * contract("Ik, k->Ik", u_temp, self.r)

        if self.has_positron:
            u_temp = contract("IJ, Jk->Ik", self.z_Omega["0"], u[-1])
            u_new[-1] -= self.laser(t) * contract("Ik, k->Ik", u_temp, self.r)

        return u_new.ravel() if ravel else u_new
