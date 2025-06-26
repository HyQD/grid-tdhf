import pytest
import subprocess
import numpy as np
from pathlib import Path

try:
    import mpi4py
except ImportError:
    pytest.skip("Parallel testing requires mpi4py", allow_module_level=True)


@pytest.mark.parametrize(
    "atom,reference_file,n_nodes",
    [
        ("be", "data/be_samples.npz", "2"),
        ("ne", "data/ne_samples.npz", "5"),
        ("psh", "data/psh_samples.npz", "2"),
    ],
)
def test_parallel_simulation(tmp_path, atom, reference_file, n_nodes):
    cmd = [
        "mpirun",
        "-np",
        n_nodes,
        "grid-tdhf-mpi",
        "-atom",
        atom,
        "-N",
        "40",
        "-r-max",
        "30",
        "-nl",
        "4",
        "-nL",
        "4",
        "-E0",
        "0.001",
        "-omega",
        "1.0",
        "-ncycles",
        "1",
        "-ncycles-after-pulse",
        "0",
        "-dt",
        "0.1",
        "-laser-name",
        "SineSquareLaser",
        "-gauge",
        "velocity",
        "-integrator-name",
        "CN",
        "-init-state",
        "scf",
        "-mask-name",
        "no-mask",
        "--output-dir",
        str(tmp_path),
        "--fileroot",
        atom,
    ]

    subprocess.run(cmd, check=True)

    samples_files = list(tmp_path.glob("*_samples.npz"))
    assert samples_files, "No output samples found!"
    output_data = np.load(samples_files[0])

    reference_data = np.load(Path(__file__).parent / reference_file)

    np.testing.assert_allclose(
        output_data["expec_z"],
        reference_data["expec_z"],
        rtol=1e-12,
        atol=1e-11,
        err_msg=f"Expec_z mismatch for atom={atom}",
    )
