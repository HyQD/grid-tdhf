class RHSDipoleVelocityGauge_3m:
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
                u[p], du_dr, self.z_Omega[f"{m}"], self.H_z_beta[f"{m}"], self.r_inv
            )

        if self.has_positron:
            du_dr = contract("ij, Ij->Ii", self.D1, u[-1])

            u_new[-1] = -self.laser(t) * pz_psi(
                u[-1], du_dr, self.z_Omega["0"], self.H_z_beta["0"], self.r_inv
            )

        if ravel:
            return u_new.ravel()
        else:
            return u_new
