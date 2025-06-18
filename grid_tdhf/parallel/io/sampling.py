from mpi4py import MPI

import numpy as np
from collections import defaultdict


class Sampler:
    required_params = {
        "comm",
        "properties_computer",
        "sample_expec_z",
        "expec_z_sample_interval",
        "sample_norm",
        "norm_sample_interval",
        "sample_energy",
        "energy_sample_interval",
        "sample_state",
        "state_sample_interval",
    }

    def __init__(
        self,
        *,
        comm,
        properties_computer,
        sample_expec_z,
        expec_z_sample_interval,
        sample_norm,
        norm_sample_interval,
        sample_energy,
        energy_sample_interval,
        sample_state,
        state_sample_interval,
    ):
        self.comm = comm

        self.properties_computer = properties_computer

        self.sample_expec_z = sample_expec_z
        self.expec_z_sample_interval = expec_z_sample_interval
        self.sample_norm = sample_norm
        self.norm_sample_interval = norm_sample_interval
        self.sample_energy = sample_energy
        self.energy_sample_interval = energy_sample_interval
        self.sample_state = sample_state
        self.state_sample_interval = state_sample_interval

        self.samples = defaultdict(list)
        self.sampled_states = []
        self.state_time_points = []

    def sample(self, state, t, count):
        comm = self.comm
        rank = comm.Get_rank()

        print("sampling", rank, t, count)

        if self.sample_expec_z and not (count % self.expec_z_sample_interval):
            local_z = self.properties_computer.compute_expec_z(state)
            global_z = comm.allgather(local_z[0])
            if rank == 0:
                self.samples["expec_z"].append(global_z)
                self.samples["expec_z_time_points"].append(t)

        if self.sample_norm and not (count % self.norm_sample_interval):
            local_norm = self.properties_computer.compute_norm(state)
            global_norm = comm.allgather(local_norm[0])
            if rank == 0:
                self.samples["norm"].append(global_norm)
                self.samples["norm_time_points"].append(t)

        if self.sample_energy and not (count % self.energy_sample_interval):
            local_energy, local_orbital_energies = (
                self.properties_computer.compute_energy(state)
            )

            global_energy = comm.allreduce(local_energy[0], op=MPI.SUM)
            global_orbital_energies = comm.allgather(local_orbital_energies[0])

            if rank == 0:
                self.samples["energy"].append(global_energy)
                self.samples["orbital_energies"].append(
                    np.concatenate(global_orbital_energies[0])
                )
                self.samples["energy_time_points"].append(t)

        if self.sample_state and not (count % self.state_sample_interval):
            full_state = gather_full_state(state, comm)
            if rank == 0:
                self.sampled_states.append(full_state.copy())
                self.state_time_points.append(t)

        if rank == 0:
            self.samples["time_points"].append(t)

    def gather_full_state(local_state, comm):
        rank = comm.Get_rank()
        size = comm.Get_size()

        nl, nr = local_state.shape[1]
        local_state.shape[2]

        if rank == 0:
            global_state = np.empty((size, nl, nr), dtype=np.complex128)
        else:
            global_state = None

        comm.Gather(local_state, global_state, root=0)
        return global_state

    def get_prepared_samples(self):
        result = {}
        for key, values in self.samples.items():
            result[key] = np.array(values)

        return result

    def get_prepared_state(self):
        result = {}
        result["state"] = np.array(self.sampled_states)
        result["state_time_points"] = np.array(self.state_time_points)

        return result
