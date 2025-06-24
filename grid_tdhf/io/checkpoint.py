from pathlib import Path
import numpy as np
import os


class CheckpointManager:
    def __init__(
        self,
        fileroot,
        sampler,
        inputs,
        full_state,
        active_orbitals,
        checkpoint_interval,
        total_steps,
        output_dir="output/",
    ):
        self.fileroot = fileroot
        self.output_dir = output_dir
        self.sampler = sampler
        self.inputs = inputs
        self.full_state = full_state
        self.active_orbitals = active_orbitals
        self.checkpoint_interval = checkpoint_interval
        self.total_steps = total_steps

        self._ensure_dir_exists()

    def checkpoint(self, current_state, current_time, current_step):
        if self.checkpoint_interval and not (current_step % self.checkpoint_interval):
            self._save_state(current_state)
            self._save_samples()
            self._save_info(
                current_step,
                current_time,
                status=f"incomplete ({100*current_step/self.total_steps:.0f}%)",
            )

    def finalize(self, final_state, final_time, final_step):
        self._save_state(final_state)
        self._save_samples()
        self._save_info(final_step, final_time, status="complete")

    def _ensure_dir_exists(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def _save_state(self, state):
        states = self.sampler.get_prepared_state()

        self.full_state[self.active_orbitals] = state
        states["u"] = self.full_state
        states["active_orbitals"] = self.active_orbitals

        self._atomic_savez(f"{self.output_dir}/{self.fileroot}_state", states)

    def _save_samples(self):
        samples = self.sampler.get_prepared_samples()
        self._atomic_savez(f"{self.output_dir}/{self.fileroot}_samples", samples)

    def _save_info(self, current_step, current_time, status):
        inputs = vars(self.inputs)
        metadata = {
            "ckpt_status": status,
            "current_step": current_step,
            "total_steps": self.total_steps,
            "current_time": current_time,
        }

        info = {
            "inputs": inputs,
            "metadata": metadata,
        }

        self._atomic_savez(f"{self.output_dir}/{self.fileroot}_info", info)

    def _atomic_savez(self, filename, arrays):
        tmp_filename = filename + ".tmp.npz"
        np.savez(tmp_filename, **arrays)
        os.replace(tmp_filename, filename + ".npz")
