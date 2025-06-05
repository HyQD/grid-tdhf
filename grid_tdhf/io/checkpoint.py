import numpy as np
import os


class CheckpointManager:
    def __init__(
        self,
        fileroot,
        sampler,
        inputs,
        checkpoint_interval,
        total_steps,
        direc="output",
    ):
        self.fileroot = fileroot
        self.direc = direc
        self.sampler = sampler
        self.inputs = inputs
        self.checkpoint_interval = checkpoint_interval
        self.total_steps = total_steps

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

    def _save_state(self, state):
        states = self.sampler.get_prepared_state()
        states["u"] = state
        self._atomic_savez(f"{self.direc}/{self.fileroot}_state", states)

    def _save_samples(self):
        samples = self.sampler.get_prepared_samples()
        self._atomic_savez(f"{self.direc}/{self.fileroot}_samples", samples)

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

        self._atomic_savez(f"{self.direc}/{self.fileroot}_info", info)

    def _atomic_savez(self, filename, arrays):
        tmp_filename = filename + ".tmp.npz"
        np.savez(tmp_filename, **arrays)
        os.replace(tmp_filename, filename + ".npz")
