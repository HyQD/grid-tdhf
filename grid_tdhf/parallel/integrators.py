import numpy as np
from mpi4py import MPI

from grid_tdhf.integrators import CN


class CNCMF2:
    required_params = {
        "comm",
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
        comm,
        N_orbs,
        nl,
        nr,
        preconditioner,
        bicgstab_tol,
        potential_computer,
        imaginary=False,
    ):
        integrator1 = CN(
            N_orbs, nl, nr, preconditioner, bicgstab_tol, imaginary=imaginary
        )

        integrator2 = CN(
            N_orbs, nl, nr, preconditioner, bicgstab_tol, imaginary=imaginary
        )

        self.integrator = _MidpointMeanFieldIntegrator(
            comm=comm,
            integrator=integrator1,
            half_step_integrator=integrator2,
            potential_computer=potential_computer,
        )

    def __call__(self, u, t, dt, rhs):
        return self.integrator(u, t, dt, rhs)


class _MidpointMeanFieldIntegrator:
    def __init__(self, comm, integrator, half_step_integrator, potential_computer):
        self.comm = comm
        self.integrator = integrator
        self.half_step_integrator = half_step_integrator
        self.potential_computer = potential_computer

    def __call__(self, u, t, dt, rhs):
        size = self.comm.Get_size()

        u_temp = self.half_step_integrator(u, t, dt / 2, rhs)

        global_u = np.zeros((size, u.shape[1], u.shape[2]), dtype=np.complex128)
        self.comm.Allgather([u_temp, MPI.COMPLEX16], [global_u, MPI.COMPLEX16])

        self.potential_computer.set_state(global_u)
        self.potential_computer.compute_direct_potential()

        u = self.integrator(u, t, dt / 2, rhs)
        u = self.integrator(u, t + dt / 2, dt / 2, rhs)

        return u
