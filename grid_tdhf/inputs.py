import argparse
from types import SimpleNamespace

from grid_tdhf.setup.load_run import load_info


def parse_arguments(verbose=True):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-atom", type=str, default="he", help="He, Be, Ne, Ar, PsH, PsCl"
    )
    parser.add_argument(
        "-charge",
        type=int,
        default=0,
        help="Charge of the system (nucleus + electron + positron). Changes the nuclear charge.",
    )
    parser.add_argument(
        "-N",
        type=int,
        default=400,
        help="",
    )
    parser.add_argument(
        "-r-max",
        "-r_max",
        dest="r_max",
        type=float,
        default=300,
        help="",
    )
    parser.add_argument(
        "-nl",
        type=int,
        default=None,
        help="",
    )
    parser.add_argument(
        "-l-max",
        "-l_max",
        dest="l_max",
        type=int,
        default=None,
        help="",
    )
    parser.add_argument(
        "-nL",
        type=int,
        default=4,
        help="",
    )
    parser.add_argument(
        "-E0",
        type=float,
        default=1e-4,
        help="",
    )
    parser.add_argument(
        "-omega",
        type=float,
        default=0.057,
        help="",
    )
    parser.add_argument(
        "-ncycles",
        type=int,
        default=1,
        help="",
    )
    parser.add_argument(
        "-ncycles-after-pulse",
        "-ncycles_after_pulse",
        dest="ncycles_after_pulse",
        type=float,
        default=0,
        help="",
    )
    parser.add_argument(
        "-ncycles_ramp",
        "-ncycles_ramp",
        dest="ncycles_ramp",
        type=float,
        default=2,
        help="",
    )
    parser.add_argument(
        "-phase",
        type=float,
        default=0,
        help="",
    )
    parser.add_argument(
        "-t0",
        type=float,
        default=0,
        help="",
    )
    parser.add_argument(
        "-dt",
        type=float,
        default=0.1,
        help="",
    )
    parser.add_argument(
        "-total-time",
        "-total_time",
        dest="total_time",
        type=float,
        default=None,
        help="",
    )
    parser.add_argument(
        "-laser-name",
        "-laser_name",
        dest="laser_name",
        type=str,
        default="SineSquareLaser",
        help="",
    )
    parser.add_argument(
        "-n-frozen-orbitals",
        "-n_frozen_orbitals",
        dest="n_frozen_orbitals",
        type=int,
        default=0,
        help="",
    )
    parser.add_argument(
        "-frozen-electrons",
        "-frozen_electrons",
        dest="frozen_electrons",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="",
    )
    parser.add_argument(
        "-frozen-positron",
        "-frozen_positron",
        dest="frozen_positron",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="",
    )
    parser.add_argument(
        "-gauge",
        type=str,
        default="velocity",
        help="",
    )
    parser.add_argument(
        "-integrator-name",
        "-integrator_name",
        dest="integrator_name",
        type=str,
        default="CN",
        help="",
    )
    parser.add_argument(
        "-preconditioner-name",
        "-preconditioner_name",
        dest="preconditioner_name",
        type=str,
        default="A1",
        help="",
    )
    parser.add_argument(
        "-checkpoint-interval",
        "-checkpoint_interval",
        dest="checkpoint_interval",
        type=int,
        default=0,
        help="n iterations between each checkpoint. 0 for no checkpointing",
    )
    parser.add_argument(
        "-init-state",
        "-init_state",
        dest="init_state",
        type=str,
        default="scf",
        help="",
    )
    parser.add_argument(
        "--save-gs",
        dest="save_gs",
        type=str2bool,
        nargs="?",
        const=True,
        help="",
    )
    parser.add_argument(
        "-n-scf-iter",
        "-n_scf_iter",
        dest="n_scf_iter",
        type=int,
        default=80,
        help="",
    )
    parser.add_argument(
        "-scf-alpha",
        "-scf_alpha",
        dest="scf_alpha",
        type=float,
        default=0.8,
        help="",
    )
    parser.add_argument(
        "-itp-integrator-name",
        "-itp_integrator_name",
        dest="itp_integrator_name",
        type=str,
        default="CN",
        help="",
    )
    parser.add_argument(
        "-itp-conv-tol",
        "-itp_conv_tol",
        dest="itp_conv_tol",
        type=float,
        default=1e-14,
        help="",
    )
    parser.add_argument(
        "-itp-dt",
        "-itp_dt",
        dest="itp_dt",
        type=float,
        default=0.1,
        help="",
    )
    parser.add_argument(
        "-max-itp-iter",
        "-max_itp_iter",
        dest="itp_max_iter",
        type=int,
        default=10000,
        help="",
    )
    parser.add_argument(
        "-bicgstab-tol",
        "-bicgstab_tol",
        dest="bicgstab_tol",
        type=float,
        default=1e-12,
        help="",
    )
    parser.add_argument(
        "-mask-name",
        "-mask_name",
        dest="mask_name",
        type=str,
        default="cosine_mask",
        help="",
    )
    parser.add_argument(
        "-mask-margin",
        "-mask_margin",
        dest="mask_margin",
        type=float,
        default=30,
        help="",
    )
    parser.add_argument(
        "-mask-n",
        "-mask_n",
        dest="mask_n",
        type=float,
        default=4,
        help="",
    )
    parser.add_argument(
        "-sample-expec-z",
        "-sample_expec_z",
        dest="sample_expec_z",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="",
    )
    parser.add_argument(
        "-expec-z-sample-interval",
        "-expec_z_sample_interval",
        dest="expec_z_sample_interval",
        type=int,
        default=1,
        help="",
    )
    parser.add_argument(
        "-sample-norm",
        "-sample_norm",
        dest="sample_norm",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="",
    )
    parser.add_argument(
        "-norm-sample-interval",
        "-norm_sample_interval",
        dest="norm_sample_interval",
        type=int,
        default=1,
        help="",
    )
    parser.add_argument(
        "-sample-energy",
        "-sample_energy",
        dest="sample_energy",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="",
    )
    parser.add_argument(
        "-energy-sample-interval",
        "-energy_sample_interval",
        dest="energy_sample_interval",
        type=int,
        default=1,
        help="",
    )
    parser.add_argument(
        "-sample-state",
        "-sample_state",
        dest="sample_state",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="",
    )
    parser.add_argument(
        "-state-sample-interval",
        "-state_sample_interval",
        dest="state_sample_interval",
        type=int,
        default=10,
        help="n iterations between each state sample. 0 for no state sampling",
    )
    parser.add_argument(
        "-comment",
        type=str,
        default="",
        help="",
    )
    parser.add_argument(
        "--gs-only",
        dest="gs_only",
        type=str2bool,
        nargs="?",
        const=True,
        help="",
    )
    parser.add_argument(
        "--load-run",
        dest="load_run",
        type=str,
        default=None,
        help="",
    )
    parser.add_argument(
        "--fileroot",
        dest="fileroot",
        type=str,
        default=None,
        help="",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        default=None,
        help="",
    )

    args = parser.parse_args()

    inputs = vars(args)

    default_nl = 6
    relation = lambda a, b: a == b + 1
    inputs["nl"], inputs["l_max"] = resolve_linked_parameters(
        inputs["nl"], inputs["l_max"], default_nl, "nl", "l_max", relation
    )

    load_run = inputs["load_run"]

    if load_run is not None:
        inputs, _ = load_info(load_run)
        inputs["load_run"] = load_run

    if verbose:
        print_inputs(inputs)

    return SimpleNamespace(**inputs)


def resolve_linked_parameters(value_a, value_b, default_a, key_a, key_b, relation):
    if value_a is not None and value_b is not None:
        if not relation(value_a, value_b):
            raise ValueError(
                f"Inconsistent values: {key_a}={value_a}, {key_b}={value_b}. Only one of them should be given."
            )
    elif value_a is not None:
        value_b = value_a - 1
    elif value_b is not None:
        value_a = value_b + 1
    else:
        value_a = default_a
        value_b = default_a - 1

    return value_a, value_b


def print_inputs(inputs):
    print("--- INPUTS ----------------")
    for key, value in zip(inputs.keys(), inputs.values()):
        print(f"{key}: {value}")

    print("---------------------------")
    print()


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "t", "yes", "1"):
        return True
    elif value.lower() in ("false", "f", "no", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")
