"""Fan et al. electronic reconstruction primitives, NOT a bulk SHG solver.

Source: doi:10.1038/s41524-026-02184-w, Table 2 and Methods Eqs. 9-14.
Only the nearest-neighbor two-Ni/two-I geometry is assembled here. The 20x20
matrix contains one-particle terms, not Coulomb interactions or a magnetic state.
"""

import json
from pathlib import Path

import numpy as np


def load_parameters(root: Path) -> dict:
    """Load the transcribed Table 2 without mixing conventions from other models."""
    return json.loads((root / "data/nii2/fan2026_parameters.json").read_text())


def hopping_kernel(parameters: dict, direction: str) -> np.ndarray:
    """Return 2x3 Ni-eg to I-p hopping matrix in Eq. 11, in eV."""
    if direction not in ("+x", "-x", "+y", "-y", "+z", "-z"):
        raise ValueError("direction must be +/-x, +/-y or +/-z")
    t1, t2, t3 = (parameters[k] for k in ("t1", "t2", "t3"))
    matrices = {
        "x": [[t1, 0, 0], [-t2, 0, 0]],
        "y": [[0, -t1, 0], [0, -t2, 0]],
        "z": [[0, 0, 0], [0, 0, t3]],
    }
    return (1 if direction[0] == "+" else -1) * np.array(
        matrices[direction[1]], complex
    )


def polarization_kernel(parameters: dict, component: str, direction: str) -> np.ndarray:
    """Transcribe Eq. 14's rectangular kernels on x/y bonds, including i/signs.

    Values retain the printed t-coefficient convention. These kernels are NOT
    calibrated length-gauge dipole moments in e Angstrom. Their physical units,
    prefactors and gauge must be reconciled before use in the SHG solver.
    """
    if direction not in ("+x", "-x", "+y", "-y") or component not in ("x", "y"):
        raise ValueError("Eq. 14 supports x/y components and +/-x,y bonds here")
    t1, t2 = parameters["t1"], parameters["t2"]
    sign = 1 if direction[0] == "+" else -1
    if component == "x" and direction[1] == "x":
        return sign * 1j * np.array([[0, t1, 0], [0, -t2, 0]], complex)
    if component == "y" and direction[1] == "y":
        return sign * 1j * np.array([[-t1, 0, 0], [-t2, 0, 0]], complex)
    return np.zeros((2, 3), complex)


def ligand_soc(lambda_ev: float) -> np.ndarray:
    """Printed Eq. 10 SOC, electron basis px/py/pz up/down; energy in eV.

    Its literal spectrum is {-lambda (twice), +lambda/2 (four times)}.
    The prose below Eq. 10 states the opposite ordering. An electron/hole or
    sign convention must be reconciled; this code does not silently flip it."""
    sx = np.array([[0, 1], [1, 0]], complex)
    sy = np.array([[0, -1j], [1j, 0]], complex)
    sz = np.diag([1, -1]).astype(complex)
    zero = np.zeros((2, 2), complex)
    return (
        lambda_ev
        / 2
        * np.block(
            [
                [zero, -1j * sz, 1j * sy],
                [1j * sz, zero, -1j * sx],
                [-1j * sy, 1j * sx, zero],
            ]
        )
    )


def nearest_cluster_one_particle(parameters: dict) -> tuple[np.ndarray, list[str]]:
    """Assemble 20 spin-orbitals for the nearest Z-bond, omitting interactions.

    Order: Ni_i(eg1 up/down,eg2 up/down), Ni_j(same), I_A(px/py/pz up/down),
    I_B(same). Local coordinates: Ni_i=(0,0), Ni_j=(-1,1), I_A=(-1,0),
    I_B=(0,1). These are local-axis bookkeeping units, not a bulk lattice.
    Ni site energy is the reference; ligand Delta follows Table 2's convention.
    """
    local = parameters["local"]
    hop = parameters["nearest_neighbor"]
    h = np.zeros((20, 20), complex)
    offsets = {"Ni_i": (0, 4), "Ni_j": (4, 8), "I_A": (8, 14), "I_B": (14, 20)}
    labels = []
    for site in offsets:
        orbitals = ("eg1", "eg2") if site.startswith("Ni") else ("px", "py", "pz")
        labels.extend(
            f"{site}:{orb}:{spin}" for orb in orbitals for spin in ("up", "down")
        )
    for start, end in (offsets["I_A"], offsets["I_B"]):
        h[start:end, start:end] = local["Delta"] * np.eye(6) + ligand_soc(
            local["lambda_I"]
        )

    def bond(a, b, matrix):
        start, end = offsets[a]
        other_start, other_end = offsets[b]
        h[start:end, other_start:other_end] += matrix
        h[other_start:other_end, start:end] += matrix.conj().T

    for ni, iodine, direction in [
        ("Ni_i", "I_A", "-x"),
        ("Ni_i", "I_B", "+y"),
        ("Ni_j", "I_B", "+x"),
        ("Ni_j", "I_A", "-y"),
    ]:
        bond(ni, iodine, np.kron(hopping_kernel(hop, direction), np.eye(2)))
    r = -np.array([[hop["r1"], hop["r3"]], [hop["r3"], hop["r2"]]])
    v = -np.array([[hop["v1"], hop["v2"], 0], [hop["v2"], hop["v1"], 0], [0, 0, 0]])
    bond("Ni_i", "Ni_j", np.kron(r, np.eye(2)))
    bond("I_A", "I_B", np.kron(v, np.eye(2)))
    return h, labels


def local_kanamori(
    orbitals: int, u_ev: float, j_ev: float
) -> tuple[np.ndarray, np.ndarray]:
    """Eq. 9 local Coulomb part in a 2^(2*orbitals) Fock basis; return H and N.

    Supports 2 or 3 orbitals. Spin-orbitals are ordered orbital-major up/down;
    bit p is the occupation of orbital p. Delta and SOC are separate one-body
    terms. This does not assemble the full cluster many-body Hamiltonian.
    """
    if orbitals not in (2, 3):
        raise ValueError("Only two- or three-orbital local terms are supported")
    size = 2 ** (2 * orbitals)
    creations = []
    for p in range(2 * orbitals):
        op = np.zeros((size, size), complex)
        for state in range(size):
            if not state & (1 << p):
                sign = (-1) ** ((state & ((1 << p) - 1)).bit_count())
                op[state | (1 << p), state] = sign
        creations.append(op)
    annihilations = [c.conj().T for c in creations]
    occupations = [c @ a for c, a in zip(creations, annihilations, strict=True)]
    h = np.zeros((size, size), complex)
    for a in range(orbitals):
        h += u_ev * occupations[2 * a] @ occupations[2 * a + 1]
        for b in range(a + 1, orbitals):
            for spin in (0, 1):
                for other_spin in (0, 1):
                    h += (
                        (u_ev - 2 * j_ev - j_ev * (spin == other_spin))
                        * occupations[2 * a + spin]
                        @ occupations[2 * b + other_spin]
                    )
        for b in range(orbitals):
            if a == b:
                continue
            h += j_ev * (
                creations[2 * a]
                @ creations[2 * a + 1]
                @ annihilations[2 * b + 1]
                @ annihilations[2 * b]
                - creations[2 * a]
                @ annihilations[2 * a + 1]
                @ creations[2 * b + 1]
                @ annihilations[2 * b]
            )
    return h, sum(occupations)
