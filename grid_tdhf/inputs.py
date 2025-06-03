import argparse
from types import SimpleNamespace


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
        default=1,
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
        type=bool,
        default=False,
        help="",
    )
    parser.add_argument(
        "-frozen-positron",
        "-frozen_positron",
        dest="frozen_positron",
        type=bool,
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
        "-n-scf-it",
        "-n_scf_it",
        dest="n_scf_it",
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
        "-BICGSTAB-tol",
        "-BICGSTAB_tol",
        dest="BICGSTAB_tol",
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
        "-mask-r0",
        "-mask_r0",
        dest="mask_r0",
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
        type=bool,
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
        type=bool,
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
        type=bool,
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
        type=bool,
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
        "-laser-name",
        "-laser_name",
        dest="laser_name",
        type=str,
        default="SineSquareLaser",
        help="",
    )

    args = parser.parse_args()

    inputs = vars(args)

    default_nl = 6
    relation = lambda a, b: a == b + 1
    inputs["nl"], inputs["l_max"] = resolve_linked_parameters(
        inputs["nl"], inputs["l_max"], default_nl, "nl", "l_max", relation
    )

    if verbose:
        print_inputs(inputs)

    return SimpleNamespace(**vars(args))


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
