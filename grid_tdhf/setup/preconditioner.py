import importlib

from grid_tdhf.utils import select_keys

from scipy.sparse.linalg import LinearOperator


def setup_preconditioner(simulation_config, imaginary=False):
    N_orbs = simulation_config.N_orbs
    nl = simulation_config.nl
    nr = simulation_config.nr

    preconditioner_name = simulation_config.preconditioner_name

    params = {**vars(simulation_config)}

    module = importlib.import_module("grid_tdhf.preconditioners")
    Preconditioner = getattr(module, preconditioner_name)

    missing_params = Preconditioner.required_params - params.keys() - {"imaginary"}
    if missing_params:
        raise ValueError(
            f"Missing required parameters for {preconditioner_name}: {', '.join(sorted(missing_params))}"
        )

    preconditioner_args = select_keys(params, Preconditioner.required_params)

    preconditioner = Preconditioner(**preconditioner_args, imaginary=imaginary)

    return LinearOperator((N_orbs * nl * nr, N_orbs * nl * nr), matvec=preconditioner)
