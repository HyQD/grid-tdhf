import numpy as np
from opt_einsum import contract


class PotentialComputer:
    def __init__(self, inputs, system_info, aux_arrays, rme):
        self.inputs = inputs
        self.system_info = system_info
        self.aux_arrays = aux_arrays
        self.rme = rme

        self.u = None
        self.V_d_electron = None
        self.V_d_positron = None
        self.V_x = None

    def setup_gm_dicts(self):
        nl = self.inputs.nl
        nL = self.inputs.nL

        n_orbs = self.system_info.n_orbs
        m_list = self.system_info.m_list
        m_max = self.system_info.m_max

        gaunt_dict = self.aux_arrays.gaunt_dict

        self.gm_dict1 = {
            m: np.zeros((n_orbs, nl, nL, nl)) for m in range(-m_max, m_max + 1)
        }
        self.gm_dict2 = {
            m: np.zeros((n_orbs, nl, nL, nl)) for m in range(-m_max, m_max + 1)
        }

        for i in range(n_orbs):
            mi = m_list[i]

            for mp in range(-m_max, m_max + 1):
                self.gm_dict1[mp][i] = (-1) ** (mi - mp) * gaunt_dict[(mp, mi)]
                self.gm_dict2[mp][i] = gaunt_dict[(mi, mp)]

    def set_state(self, u):
        self.u = u.copy()

    def construct_potentials(self, u):
        self.construct_direct_potential()
        self.construct_exchange_potential(u)

    def construct_direct_potential(self):
        nl = self.inputs.nl

        n_orbs = self.system_info.n_orbs
        m_list = self.system_info.m_list
        m_max = self.system_info.m_max
        has_positron = self.system_info.has_positron

        poisson_inverse = self.aux_arrays.poisson_inverse
        gaunt_dict = self.aux_arrays.gaunt_dict

        nr = self.rme.nr

        self.V_d_electron = {
            m: np.zeros((nl, nl, nr), dtype=np.complex128)
            for m in range(-m_max, m_max + 1)
        }

        if has_positron:
            self.V_d_positron = {
                m: np.zeros((nl, nl, nr), dtype=np.complex128)
                for m in range(-m_max, m_max + 1)
            }
        else:
            self.V_d_positron = None

        for mp in range(-m_max, m_max + 1):
            g_mat2 = gaunt_dict[(mp, mp)]
            for j in range(n_orbs):
                mj = m_list[j]
                g_mat1 = gaunt_dict[(mj, mj)]
                self.V_d_electron[mp] += (
                    4
                    * np.pi
                    * contract(
                        "Lrs,ms,ns,mLn,oLl->olr",
                        poisson_inverse,
                        self.u[j].conj(),
                        self.u[j],
                        g_mat1,
                        g_mat2,
                    )
                )

            if has_positron:
                g_mat = gaunt_dict[(0, 0)]
                self.V_d_positron[mp] = (
                    4
                    * np.pi
                    * contract(
                        "Lrs,ms,ns,mLn,oLl->olr",
                        poisson_inverse,
                        self.u[-1].conj(),
                        self.u[-1],
                        g_mat,
                        g_mat2,
                    )
                )

    def construct_exchange_potential(self, u):
        nl = self.inputs.nl
        n_frozen_orbitals = self.inputs.n_frozen_orbitals

        n_orbs = self.system_info.n_orbs
        m_list = self.system_info.m_list

        nr = self.rme.nr

        poisson_inverse = self.aux_arrays.poisson_inverse
        gaunt_dict = self.aux_arrays.gaunt_dict

        self.V_x = np.zeros((n_orbs, n_orbs, nl, nl, nr), dtype=np.complex128)

        for j in range(n_orbs):
            for p in range(n_frozen_orbitals, n_orbs):
                mj = m_list[j]
                mp = m_list[p]
                g_mat1 = gaunt_dict[(mj, mp)]
                g_mat2 = gaunt_dict[(mp, mj)]
                self.V_x[j, p, :, :, :] = (
                    4
                    * (-1) ** (mj - mp)
                    * np.pi
                    * contract(
                        "Lsr, mr, nr, mLn, oLl -> ols",
                        poisson_inverse,
                        self.u[j].conj(),
                        u[p],
                        g_mat1,
                        g_mat2,
                    )
                )
