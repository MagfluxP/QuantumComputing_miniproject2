"""Small-instance coherent optical response with exact spectral resolvent blocks.

Material branch uses dense dilations of resolvents and a commutator in Liouville
space. It is an explicit quantum circuit, not the toy QSP phase-filter pipeline.
Classically constructing these small matrices supplies no scalable oracle or
quantum advantage. The original QSP implementation remains in qsp.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, Statevector


def dilation(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """One-ancilla unitary block encoding A/||A|| for any square contraction."""
    scale = max(float(np.linalg.norm(matrix, 2)), 1e-15)
    a = matrix / scale
    if np.count_nonzero(a - np.diag(np.diag(a))) == 0:
        root = np.diag(np.sqrt(np.clip(1 - abs(np.diag(a)) ** 2, 0, None)))
        return np.block([[a, root], [root, -a.conj().T]]), scale
    u, s, vh = np.linalg.svd(a)
    left = (u * np.sqrt(np.clip(1 - s * s, 0, None))) @ u.conj().T
    v = vh.conj().T
    right = (v * np.sqrt(np.clip(1 - s * s, 0, None))) @ v.conj().T
    unitary = np.block([[a, left], [right, -a.conj().T]])
    return unitary, scale


def state_preparation_unitary(vector: np.ndarray) -> np.ndarray:
    """Stable dense Householder preparation, including nearly zero amplitudes.

    This is small-instance state loading, not an efficient scalable oracle.
    """
    vector = np.asarray(vector, dtype=complex)
    vector = vector / np.linalg.norm(vector)
    phase = np.exp(1j * np.angle(vector[0]))
    difference = vector.copy()
    difference[0] -= phase
    norm = np.vdot(difference, difference).real
    unitary = np.eye(len(vector), dtype=complex)
    if norm > 1e-28:
        unitary -= 2 * np.outer(difference, difference.conj()) / norm
    unitary[:, 0] *= phase
    return unitary


def quantum_response(
    energies: np.ndarray,
    mu: np.ndarray,
    photon_ev: float,
    gamma_ev: float,
    hadamard: bool = True,
    output_mu: np.ndarray | None = None,
    initial_rho: np.ndarray | None = None,
    second_gamma_factor: float = 1.0,
) -> dict[str, Any]:
    """Return complex beta from accepted circuit overlap and optional readout.

    Inputs: energies (K,), Hermitian dipoles/density (K,K), power-of-two K,
    gamma_ev>0; output dictionary contains beta, scales, circuit and readout.
    Column-major vec(rho); C=I kron mu - mu.T kron I; resolvent R_k diagonal.
    beta=<vec(mu)| R2 C R1 C |vec(rho0)>; two minus signs cancel.
    All blocks retain normalization and fresh ancillas. No intermediate readout.
    """
    k = len(energies)
    if k < 2 or k & (k - 1) or gamma_ev <= 0:
        raise ValueError("Use a power-of-two state count and positive broadening")
    identity = np.eye(k)
    commutator = np.kron(identity, mu) - np.kron(mu.T, identity)
    gaps = (energies[:, None] - energies[None, :]).reshape(-1, order="F")
    r1 = np.diag(1 / (photon_ev - gaps + 1j * gamma_ev))
    r2 = np.diag(1 / (2 * photon_ev - gaps + 1j * gamma_ev * second_gamma_factor))
    matrices = [commutator, r1, commutator, r2]
    blocks = [dilation(m) for m in matrices]
    system = 2 * int(np.log2(k))
    qc = QuantumCircuit(system + 4)
    targets = list(range(system))
    initial_preparation = None
    rho_norm = 1.0
    if initial_rho is not None:
        if np.asarray(initial_rho).shape != (k, k):
            raise ValueError("initial density shape mismatch")
        rho_norm = float(np.linalg.norm(initial_rho))
        if rho_norm < 1e-14:
            raise ValueError("Zero initial density")
        initial_preparation = UnitaryGate(
            state_preparation_unitary(
                np.asarray(initial_rho).reshape(-1, order="F") / rho_norm
            ),
            label="rho prep",
        )
        qc.append(initial_preparation, targets)
    for index, (unitary, _) in enumerate(blocks):
        qc.append(
            UnitaryGate(unitary, label=["C", "R1", "C", "R2"][index]),
            targets + [system + index],
        )
    readout_mu = mu if output_mu is None else output_mu
    vector = readout_mu.reshape(-1, order="F")
    norm = np.linalg.norm(vector)
    if norm < 1e-14:
        raise ValueError("Dipole readout vector has zero norm")
    prep = UnitaryGate(state_preparation_unitary(vector / norm), label="mu prep")
    qc.append(prep.inverse(), targets)
    scale = float(norm * rho_norm * np.prod([s for _, s in blocks]))
    beta = complex(Statevector.from_instruction(qc).data[0] * scale)
    result = {
        "beta": beta,
        "scale": scale,
        "circuit": qc,
        "system_qubits": system,
        "work_qubits": 4,
        "hadamard_beta": None,
        "hadamard_probabilities": {},
    }
    if hadamard:
        readout = np.asarray(Operator(prep.inverse()).data)
        quadratures = []
        for quadrature in ("real", "imag"):
            test = QuantumCircuit(system + 5)
            control = system + 4
            test.h(control)
            if initial_preparation is not None:
                initial_matrix = np.asarray(Operator(initial_preparation).data)
                controlled_initial = np.eye(2 * len(initial_matrix), dtype=complex)
                controlled_initial[1::2, 1::2] = initial_matrix
                test.append(UnitaryGate(controlled_initial), [control] + targets)
            for index, (unitary, _) in enumerate(blocks):
                controlled = np.eye(2 * len(unitary), dtype=complex)
                controlled[1::2, 1::2] = unitary
                test.append(
                    UnitaryGate(controlled), [control] + targets + [system + index]
                )
            controlled = np.eye(2 * len(readout), dtype=complex)
            controlled[1::2, 1::2] = readout
            test.append(UnitaryGate(controlled), [control] + targets)
            if quadrature == "imag":
                test.sdg(control)
            test.h(control)
            probabilities = Statevector.from_instruction(test).probabilities([control])
            result["hadamard_probabilities"][quadrature] = probabilities.tolist()
            quadratures.append(scale * (probabilities[0] - probabilities[1]))
        result["hadamard_beta"] = complex(*quadratures)
    return result
