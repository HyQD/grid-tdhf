import numpy as np


class A1:
    required_params = {"D2", "r", "nl", "nr", "n_orbs", "dt", "imaginary"}

    def __init__(self, *, D2, r, nl, nr, n_orbs, dt, imaginary=False):
        self.nl = nl
        self.nr = nr
        self.n_orbs = n_orbs

        Identity = np.complex128(np.eye(self.nr))
        self.M_l = np.zeros((self.nl, self.nr, self.nr), dtype=np.complex128)

        time_factor = 1 if imaginary else 1j

        for l in range(self.nl):
            T_l = -(1 / 2) * D2 + np.diag(l * (l + 1) / (2 * r**2))
            self.M_l[l] = np.linalg.inv(Identity + time_factor * dt / 2 * T_l)

        self.N_orbs = self.n_orbs

    def __call__(self, psi):
        psi = psi.reshape((self.N_orbs, self.nl, self.nr))
        psi_new = np.zeros((self.N_orbs, self.nl, self.nr), dtype=np.complex128)

        for i in range(self.N_orbs):
            for l in range(self.nl):
                psi_new[i, l] = np.dot(self.M_l[l], psi[i, l])

        return psi_new.ravel()
