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
        "-r_max",
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
        "-l_max",
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
        "-ncycles_after_pulse",
        type=float,
        default=1,
        help="",
    )
    parser.add_argument(
        "-ncycles_ramp",
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
        "-n_frozen_orbitals",
        type=int,
        default=0,
        help="",
    )
    parser.add_argument(
        "-frozen_electrons",
        type=bool,
        default=False,
        help="",
    )
    parser.add_argument(
        "-frozen_positron",
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
        "-integrator",
        type=str,
        default="CN",
        help="",
    )
    parser.add_argument(
        "-preconditioner",
        type=str,
        default="A1",
        help="",
    )
    parser.add_argument(
        "-ckpt_freq",
        type=int,
        default=0,
        help="n iterations between each checkpoint. 0 for no checkpointing",
    )
    parser.add_argument(
        "-ckpt_name",
        type=str,
        default=None,
        help="",
    )
    parser.add_argument(
        "-sample_wf_freq",
        type=int,
        default=0,
        help="n iterations between each wavefunction sample. 0 for no wf sampling",
    )
    parser.add_argument(
        "-exchange_type",
        type=int,
        default=2,
        help="1: llrr, 2: ppllr",
    )
    parser.add_argument(
        "-init_state",
        type=str,
        default="scf",
        help="",
    )
    parser.add_argument(
        "-scf_n_it",
        type=int,
        default=80,
        help="",
    )
    parser.add_argument(
        "-scf_alpha",
        type=float,
        default=0.8,
        help="",
    )
    parser.add_argument(
        "-BICGSTAB_tol",
        type=float,
        default=1e-12,
        help="",
    )
    parser.add_argument(
        "-use_mask",
        type=bool,
        default=True,
        help="",
    )
    parser.add_argument(
        "-compute_energy",
        type=bool,
        default=True,
        help="",
    )
    parser.add_argument(
        "-comment",
        type=str,
        default="",
        help="",
    )
    parser.add_argument(
        "-laser_class",
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
