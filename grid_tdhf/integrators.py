from grid_tdhf.exceptions import ConvergenceError
from grid_lib.spherical_coordinates.utils import Counter

from scipy.sparse.linalg import LinearOperator, bicgstab
import scipy
import numpy as np

from packaging import version

SCIPY_VERSION = version.parse(scipy.__version__)


class IMPCDM1:
    required_params = {
        "N_orbs",
        "nl",
        "nr",
        "preconditioner",
        "bicgstab_tol",
        "imaginary",
    }

    def __init__(self, N_orbs, nl, nr, preconditioner, bicgstab_tol, imaginary=False):
        self.N_orbs = N_orbs
        self.nl = nl
        self.nr = nr
        self.preconditioner = preconditioner
        self.time_factor = 1 if imaginary else 1j

        if SCIPY_VERSION >= version.parse("1.12.0"):
            self.tol_kwargs = {"rtol": bicgstab_tol, "atol": 0.0}
        else:
            self.tol_kwargs = {"tol": bicgstab_tol}

    def __call__(self, u, t, dt, rhs):
        ti = t + dt / 2

        time_factor = self.time_factor
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
            callback=local_counter,
            **self.tol_kwargs,
        )

        if info != 0:
            raise ConvergenceError("BICGSTAB did not converge")

        u = u.reshape((N_orbs, nl, nr))

        return u


class IMPCDM2:
    required_params = {
        "N_orbs",
        "nl",
        "nr",
        "preconditioner",
        "bicgstab_tol",
        "potential_computer",
        "imaginary",
    }

    def __init__(
        self,
        N_orbs,
        nl,
        nr,
        preconditioner,
        bicgstab_tol,
        potential_computer,
        imaginary=False,
    ):
        integrator1 = IMPCDM1(
            N_orbs, nl, nr, preconditioner, bicgstab_tol, imaginary=imaginary
        )

        integrator2 = IMPCDM1(
            N_orbs, nl, nr, preconditioner, bicgstab_tol, imaginary=imaginary
        )

        self.integrator = _MidpointMeanFieldIntegrator(
            integrator=integrator1,
            half_step_integrator=integrator2,
            potential_computer=potential_computer,
        )

    def __call__(self, u, t, dt, rhs):
        return self.integrator(u, t, dt, rhs)


class IMPSCF:
    required_params = {
        "N_orbs",
        "nl",
        "nr",
        "preconditioner",
        "m_list",
        "has_positron",
        "weights",
        "bicgstab_tol",
        "midpoint_scf_tol",
        "potential_computer",
        "imaginary",
    }

    def __init__(
        self,
        N_orbs,
        nl,
        nr,
        preconditioner,
        m_list,
        has_positron,
        weights,
        bicgstab_tol,
        midpoint_scf_tol,
        potential_computer,
        imaginary=False,
    ):
        integrator = IMPCDM1(
            N_orbs, nl, nr, preconditioner, bicgstab_tol, imaginary=imaginary
        )

        self.integrator = _SelfConsistentMidpointMeanFieldIntegrator(
            integrator=integrator,
            potential_computer=potential_computer,
            m_list=m_list,
            has_positron=has_positron,
            weights=weights,
            midpoint_scf_tol=midpoint_scf_tol,
        )

    def __call__(self, u, t, dt, rhs):
        return self.integrator(u, t, dt, rhs)


class _MidpointMeanFieldIntegrator:
    def __init__(self, integrator, half_step_integrator, potential_computer):
        self.integrator = integrator
        self.half_step_integrator = half_step_integrator
        self.potential_computer = potential_computer

    def __call__(self, u, t, dt, rhs):
        u_temp = self.half_step_integrator(u, t, dt / 2, rhs)

        self.potential_computer.set_state(u_temp)
        self.potential_computer.compute_direct_potential()

        u = self.integrator(u, t, dt / 2, rhs)
        u = self.integrator(u, t + dt / 2, dt / 2, rhs)

        return u


class _SelfConsistentMidpointMeanFieldIntegrator:
    def __init__(
        self,
        integrator,
        potential_computer,
        m_list,
        has_positron,
        weights,
        midpoint_scf_tol,
    ):
        self.integrator = integrator
        self.potential_computer = potential_computer

        self.m_list = m_list
        self.has_positron = has_positron
        self.weights = weights
        self.midpoint_scf_tol = midpoint_scf_tol

    def __call__(self, u, t, dt, rhs):
        midpoint_scf_diff = np.inf

        u_midpoint_old = u.copy()
        u_midpoint = u.copy()

        while midpoint_scf_diff > self.midpoint_scf_tol:
            self.potential_computer.set_state(u_midpoint)
            self.potential_computer.compute_direct_potential()

            u_new = self.integrator(u_midpoint, t, dt, rhs)

            u_midpoint = (u_new + u) / 2

            midpoint_scf_diff = np.max(np.abs(u_midpoint - u_midpoint_old))

            u_midpoint_old = u_midpoint.copy()

        self.potential_computer.set_state(u_midpoint)
        self.potential_computer.compute_direct_potential()

        u = self.integrator(u, t, dt, rhs)

        return u
