import numpy as np
from collections import defaultdict


class Sampler:
    required_params = {
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
        self.time_points = []
        self.state_time_points = []

    def sample(self, state, t, count):
        if self.sample_expec_z and not (count % self.expec_z_sample_interval):
            expec_z = self.properties_computer.compute_expec_z(state)
            self.samples["expec_z"].append(expec_z)
            self.samples["expec_z_time_points"].append(t)

        if self.sample_norm and not (count % self.norm_sample_interval):
            norm = self.properties_computer.compute_norm(state)
            self.samples["norm"].append(norm)
            self.samples["norm_time_points"].append(t)

        if self.sample_energy and not (count % self.energy_sample_interval):
            energy, orbital_energies = self.properties_computer.compute_energy(state)
            self.samples["energy"].append(energy)
            self.samples["orbital_energies"].append(orbital_energies)
            self.samples["energy_time_points"].append(t)

        if self.sample_state and not (count % self.state_sample_interval):
            self.sampled_states.append(state.copy())
            self.state_time_points.append(t)

        self.time_points.append(t)

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
