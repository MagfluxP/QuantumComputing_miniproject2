"""Gao effective S=1 Hamiltonian on an explicit finite triangular cluster.

This is a low-energy spin model, not an electronic optical Hamiltonian. Table
S1 caption defines the adopted reference axes; its conflict with surrounding
'cubic' wording and main-text J2 sign are recorded rather than resolved silently.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector
from scipy import sparse
from scipy.linalg import expm

from .consistent_spiral import spin_coherent
from .hamiltonian_audit import spin_operators
from .response_circuit import state_preparation_unitary

HBAR_EV_FS = 0.6582119569


@dataclass(frozen=True)
class GaoSpinConfig:
    j2_source: str = "supplement"
    single_ion_factor: float = 0.5


DEFAULT_CONFIG = GaoSpinConfig()


class MissingGaoOpticalOperator(RuntimeError):
    """A spin Hamiltonian cannot alone provide an optical-frequency SHG tensor."""


def require_optical_shg_inputs() -> None:
    raise MissingGaoOpticalOperator(
        "Gao spin H0 is implemented, but a matched frequency-dependent optical SHG operator is missing. Do not reuse Fan electronic dipoles or treat Gao static fitted chi as an independent quantum prediction."
    )


def geometry() -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    # Axial triangular coordinates. Reference NN bond is along +y, as Table S1.
    axial = np.array([[0, 0], [1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1]], int)
    lattice = np.array([[0, np.sqrt(3) / 2], [1, 0.5]])
    xy = axial @ lattice.T
    bonds = []
    for i, j in combinations(range(7), 2):
        delta = axial[j] - axial[i]
        distance = int(delta[0] ** 2 + delta[1] ** 2 + delta[0] * delta[1])
        shell = {1: 1, 3: 2, 4: 3}[distance]
        vector = xy[j] - xy[i]
        direction = float(np.arctan2(vector[1], vector[0]))
        # Three UNDIRECTED bond axes. The table has no xz/yz terms, so reversing
        # a bond by pi leaves its rank-two exchange tensor unchanged.
        candidates = np.pi / 2 + np.arange(3) * 2 * np.pi / 3
        errors = abs(np.angle(np.exp(2j * (direction - candidates))))
        rotation = float(candidates[np.argmin(errors)] - np.pi / 2)
        bonds.append({"i": i, "j": j, "shell": shell, "rotation_rad": rotation})
    return axial, xy, bonds


def exchange_tensor(
    parameters: dict[str, Any],
    shell: int,
    rotation: float,
    config: GaoSpinConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    spin = parameters["spin_model"]
    anis = parameters["spin_anisotropy_meV"]
    if config.j2_source not in ["main_text", "supplement"]:
        raise ValueError("Choose main_text or supplement J2 explicitly")
    j2 = (
        spin["J2_meV"]
        if config.j2_source == "main_text"
        else spin["supplementary_J2_meV"]
    )
    iso = {1: spin["J1_meV"], 2: j2, 3: spin["J3_meV"]}[shell]
    tensor = iso * np.eye(3)
    if shell in [1, 3]:
        prefix = f"J{shell}r_"
        ref = np.diag([anis[prefix + k] for k in ["xx", "yy", "zz"]])
        ref[0, 1] = ref[1, 0] = anis[prefix + "xy"]
        c, s = np.cos(rotation), np.sin(rotation)
        r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        tensor += r @ ref @ r.T
    return tensor * 1e-3


def embed_qutrit(
    local: np.ndarray, sites: Sequence[int], count: int
) -> sparse.csr_matrix:
    """Embed a local operator; site 0 is least significant base-three digit."""
    sites = tuple(sites)
    dimension = 3**count
    if local.shape != (3 ** len(sites),) * 2:
        raise ValueError("Local matrix size mismatch")
    cols_by_local = [np.flatnonzero(local[:, col]) for col in range(local.shape[1])]
    rows = []
    cols = []
    values = []
    for col in range(dimension):
        digits = [col // 3**site % 3 for site in sites]
        local_col = sum(digit * 3**i for i, digit in enumerate(digits))
        remainder = col - sum(
            digit * 3**site for site, digit in zip(sites, digits, strict=True)
        )
        for local_row in cols_by_local[local_col]:
            row = remainder + sum(
                (int(local_row) // 3**i % 3) * 3**site for i, site in enumerate(sites)
            )
            rows.append(row)
            cols.append(col)
            values.append(local[local_row, local_col])
    return sparse.csr_matrix(
        (values, (rows, cols)), shape=(dimension, dimension), dtype=complex
    )


def build_gao_h0(root: Path, config: GaoSpinConfig = DEFAULT_CONFIG) -> dict[str, Any]:
    """Return sparse eV Hamiltonian and seven-site S=1 metadata from Gao parameters. Inputs: project root and explicit source-convention config; open finite boundaries."""
    parameters = json.loads((root / "data/nii2/gao2024_parameters.json").read_text())
    axial, xy, bonds = geometry()
    sp = spin_operators()
    dot = sum(np.kron(s, s) for s in sp)
    terms = []
    h = sparse.csr_matrix((3**7, 3**7), dtype=complex)
    for bond in bonds:
        tensor = exchange_tensor(
            parameters, bond["shell"], bond["rotation_rad"], config
        )
        # Local little-endian order: index=mi+3*mj, hence kron(Sj,Si).
        local = sum(
            tensor[a, b] * np.kron(sp[b], sp[a]) for a in range(3) for b in range(3)
        )
        if bond["shell"] == 1:
            local += parameters["spin_model"]["B_meV"] * 1e-3 * (dot @ dot)
        terms.append(
            {
                "sites": [bond["i"], bond["j"]],
                "matrix_ev": local,
                "kind": "bond",
                "shell": bond["shell"],
            }
        )
        h += embed_qutrit(local, [bond["i"], bond["j"]], 7)
    local = (
        config.single_ion_factor
        * parameters["spin_anisotropy_meV"]["Azz"]
        * 1e-3
        * (sp[2] @ sp[2])
    )
    for site in range(7):
        terms.append({"sites": [site], "matrix_ev": local, "kind": "single-ion"})
        h += embed_qutrit(local, [site], 7)
    h.eliminate_zeros()
    return {
        "h0_ev": h,
        "terms": terms,
        "positions": xy,
        "axial": axial,
        "bonds": bonds,
        "parameters": parameters,
        "config": asdict(config),
        "dimension": 3**7,
        "sites": 7,
    }


def encoded_indices(count: int) -> np.ndarray:
    indices = np.arange(3**count)
    return sum((indices // 3**site % 3) * 4**site for site in range(count))


def encode_local_hamiltonian(local: np.ndarray) -> np.ndarray:
    count = round(np.log(local.shape[0]) / np.log(3))
    indices = encoded_indices(count)
    out = np.zeros((4**count, 4**count), complex)
    out[np.ix_(indices, indices)] = local
    return out


def initial_product_state(positions: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    # A stated circuit witness, not a finite-cluster ground state or fitted domain.
    normal = np.array([-np.sin(np.deg2rad(55)), 0, np.cos(np.deg2rad(55))])
    u = np.array([0.0, 1, 0])
    v = np.cross(normal, u)
    states = []
    for x, _ in positions:
        phase = 2 * np.pi * x / 7
        states.append(spin_coherent(u * np.cos(phase) + v * np.sin(phase)))
    vector = np.array([1.0 + 0j])
    for state in reversed(states):
        vector = np.kron(vector, state)
    return vector, states


def quantum_spin_evolution(
    cluster: dict[str, Any], time_fs: float = 50.0, steps: int = 4
) -> dict[str, Any]:
    """Second-order product-formula circuit on 14 qubits, qutrit code 00,01,10.

    Dense gates are only one/two-spin (4x4/16x16); no full-space exact unitary
    or classical final-state preparation is used. |11> levels decouple.
    """
    if steps < 1 or time_fs < 0:
        raise ValueError("Require positive steps and nonnegative time")
    _, states = initial_product_state(cluster["positions"])
    qc = QuantumCircuit(14)
    for site, state in enumerate(states):
        padded = np.r_[state, 0.0]
        qc.append(
            UnitaryGate(state_preparation_unitary(padded), label="spin prep"),
            [2 * site, 2 * site + 1],
        )
    tau = time_fs / HBAR_EV_FS / steps
    gates = []
    for term in cluster["terms"]:
        unitary = expm(-0.5j * tau * encode_local_hamiltonian(term["matrix_ev"]))
        gates.append(
            (
                UnitaryGate(unitary, label=term["kind"]),
                [q for site in term["sites"] for q in [2 * site, 2 * site + 1]],
            )
        )
    for _ in range(steps):
        for gate, targets in gates + list(reversed(gates)):
            qc.append(gate, targets)
    encoded = Statevector.from_instruction(qc).data
    physical = encoded[encoded_indices(7)]
    return {
        "physical_state": physical,
        "leakage_probability": max(0.0, float(1 - np.vdot(physical, physical).real)),
        "circuit": qc,
    }
