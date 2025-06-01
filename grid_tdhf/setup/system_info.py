from dataclasses import dataclass
from typing import List, Any


@dataclass
class SystemInfo:
    N_orbs: int
    n_orbs: int
    l_list: List[int]
    m_list: List[int]
    m_max: int
    Z: int
    has_positron: bool
    orb_label: str


def get_atomic_system_params(inputs):
    atom = inputs.atom
    charge = inputs.charge

    atom = atom.lower()

    if atom == "he":
        N_orbs = 1
        n_orbs = 1
        l_list = [0]
        m_list = [0]
        Z = 2
        has_positron = False
        orb_label = "0"

    elif atom == "be":
        N_orbs = 2
        n_orbs = 2
        l_list = [0, 0]
        m_list = [0, 0]
        Z = 4
        has_positron = False
        orb_label = "1m"

    elif atom == "ne":
        N_orbs = 5
        n_orbs = 5
        l_list = [0, 0, 1, 1, 1]
        m_list = [0, 0, -1, 0, 1]
        Z = 10
        has_positron = False
        orb_label = "3m"

    elif atom == "ar":
        N_orbs = 9
        n_orbs = 9
        l_list = [0, 0, 0, 1, 1, 1, 1, 1, 1]
        m_list = [0, 0, 0, -1, 0, 1, -1, 0, 1]
        Z = 18
        has_positron = False
        orb_label = "3m"

    elif atom == "psh":
        N_orbs = 2
        n_orbs = 1
        l_list = [0, 0]
        m_list = [0, 0]
        Z = 1
        has_positron = True
        orb_label = "0"

    elif atom == "psli":
        N_orbs = 3
        n_orbs = 2
        l_list = [0, 0, 0]
        m_list = [0, 0, 0]
        Z = 3
        has_positron = True
        orb_label = "1m"

    elif atom == "psf":
        N_orbs = 6
        n_orbs = 5
        l_list = [0, 0, 1, 1, 1, 0]
        m_list = [0, 0, -1, 0, 1, 0]
        Z = 9
        has_positron = True
        orb_label = "3m"

    elif atom == "pscl":
        N_orbs = 10
        n_orbs = 9
        l_list = [0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
        m_list = [0, 0, 0, -1, 0, 1, -1, 0, 1, 0]
        Z = 17
        has_positron = True
        orb_label = "3m"

    else:
        raise NotImplementedError(f"{atom} is not implemented")

    m_max = max(m_list)
    Z += charge

    return SystemInfo(
        N_orbs=N_orbs,
        n_orbs=n_orbs,
        l_list=l_list,
        m_list=m_list,
        m_max=m_max,
        Z=Z,
        has_positron=has_positron,
        orb_label=orb_label,
    )
