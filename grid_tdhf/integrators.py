from grid_tdhf.exceptions import ConvergenceError
from grid_methods.spherical_coordinates.utils import Counter

from scipy.sparse.linalg import LinearOperator, bicgstab


class CN:
    required_params = {
        "N_orbs",
        "nl",
        "nr",
        "preconditioner",
        "BICGSTAB_tol",
        "imaginary",
    }

    def __init__(self, N_orbs, nl, nr, preconditioner, BICGSTAB_tol, imaginary=False):
        self.N_orbs = N_orbs
        self.nl = nl
        self.nr = nr
        self.tol = BICGSTAB_tol
        self.preconditioner = preconditioner
        self.time_factor = 1 if imaginary else 1j

    def __call__(self, u, t, dt, rhs):
        ti = t + dt / 2

        time_factor = self.time_factor
        tol = self.tol
        preconditioner = self.preconditioner

        N_orbs = self.N_orbs
        nl = self.nl
        nr = self.nr

        Ap_lambda = lambda u, ti=ti: u.ravel() + time_factor * dt / 2 * rhs(u, ti)
        Ap_linear = LinearOperator(
            (N_orbs * nl * nr, N_orbs * nl * nr),
            matvec=Ap_lambda,
        )

        z = u.ravel() - time_factor * dt / 2 * rhs(u, ti)

        local_counter = Counter()
        u, info = bicgstab(
            Ap_linear,
            z,
            M=preconditioner,
            x0=u.ravel(),
            rtol=tol,
            atol=tol,
            callback=local_counter,
        )

        if info != 0:
            raise ConvergenceError("BICGSTAB did not converge")

        u = u.reshape((N_orbs, nl, nr))

        return u
