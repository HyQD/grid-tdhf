import importlib

from grid_tdhf.utils import resolve_required_params

from scipy.sparse.linalg import LinearOperator


def setup_preconditioner(
    simulation_config, imaginary=False, used_inputs=None, param_mapping=None
):
    N_orbs = simulation_config.N_orbs
    nl = simulation_config.nl
    nr = simulation_config.nr

    preconditioner_name = simulation_config.preconditioner_name

    params = {**vars(simulation_config)}

    module = importlib.import_module("grid_tdhf.preconditioners")
    Preconditioner = getattr(module, preconditioner_name)

    preconditioner_args = resolve_required_params(
        Preconditioner.required_params, params, used_inputs, param_mapping
    )

    preconditioner = Preconditioner(**preconditioner_args, imaginary=imaginary)

    return LinearOperator((N_orbs * nl * nr, N_orbs * nl * nr), matvec=preconditioner)
