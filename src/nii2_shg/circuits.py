"""Explicit LCU of Chebyshev-walk powers: a small-instance polynomial filter.

This is a variant, NOT the paper's phase-sequence QSP synthesis. It retains
coherent dipole/filter overlaps but does not reproduce the paper's cost bounds.
"""

from dataclasses import dataclass
from math import ceil, log2

import numpy as np
from numpy.polynomial import Polynomial
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation, UnitaryGate
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector

from .model import Array, OpticalModel


@dataclass
class Block:
    """Unitary with <0_anc|U|0_anc>=target/scale; system is least significant."""

    circuit: QuantumCircuit
    scale: float
    system_qubits: int

    @property
    def ancillas(self) -> int:
        return self.circuit.num_qubits - self.system_qubits

    def matrix(self) -> Array:
        """Return small unitary for simulation and controlled embedding."""
        return np.asarray(Operator(self.circuit).data)

    def accepted(self) -> Array:
        """Return accepted block, including its normalization."""
        size = 2**self.system_qubits
        return self.matrix()[:size, :size]


def lcu(coefficients: list[complex], unitaries: list[Array]) -> Block:
    """Build PREP-dagger SELECT PREP, keeping signs/phases in SELECT."""
    pairs = [
        (c, u) for c, u in zip(coefficients, unitaries, strict=True) if abs(c) > 1e-13
    ]
    if not pairs:
        raise ValueError("LCU needs a nonzero coefficient")
    dimension = pairs[0][1].shape[0]
    system_qubits = round(log2(dimension))
    if 2**system_qubits != dimension:
        raise ValueError("Unitary dimension must be a power of two")
    index_qubits = max(1, ceil(log2(len(pairs))))
    scale = float(sum(abs(c) for c, _ in pairs))
    weights = np.zeros(2**index_qubits, complex)
    select = np.eye(dimension * len(weights), dtype=complex)
    for j, (coefficient, unitary) in enumerate(pairs):
        weights[j] = np.sqrt(abs(coefficient) / scale)
        select[
            j * dimension : (j + 1) * dimension, j * dimension : (j + 1) * dimension
        ] = coefficient / abs(coefficient) * unitary
    circuit = QuantumCircuit(system_qubits + index_qubits)
    indices = list(range(system_qubits, circuit.num_qubits))
    preparation = StatePreparation(weights)
    circuit.append(preparation, indices)
    circuit.append(UnitaryGate(select, label="SELECT"), range(circuit.num_qubits))
    circuit.append(preparation.inverse(), indices)
    return Block(circuit, scale, system_qubits)


def dipole_block(model: OpticalModel) -> Block:
    """Expand the supplied dipole matrix in Pauli strings and construct its LCU."""
    paulis = SparsePauliOp.from_operator(Operator(model.dipole)).simplify()
    return lcu(list(paulis.coeffs), [p.to_matrix() for p in paulis.paulis])


def hamiltonian_walk(model: OpticalModel) -> Array:
    """Return W=Z_anc U_H; accepted block of W^k is T_k(H_scaled).

    Explicit small-matrix Hermitian dilation of a diagonal toy Hamiltonian;
    this is not a scalable electronic-structure oracle.
    """
    h = model.scaled_hamiltonian
    root = np.diag(np.sqrt(np.clip(1 - np.diag(h).real ** 2, 0, None)))
    return np.block([[h, root], [-root, h]])


def filter_block(model: OpticalModel, state: int, leakage: float = 0.0) -> Block:
    """Degree-two interpolation on three active eigenvalues.

    leakage in [0,1] replaces p by (1-leakage)*p+leakage. Padding can have a
    nonzero filter value because it is optically decoupled. This measures
    pass/stop leakage, not convergence of phase-sequence QSP synthesis.
    """
    if state not in (0, 1, 2) or not 0 <= leakage <= 1:
        raise ValueError("State must be 0,1,2 and leakage must be in [0,1]")
    nodes = np.diag(model.scaled_hamiltonian).real[:3]
    polynomial = Polynomial([1.0])
    for other in range(3):
        if other != state:
            polynomial *= Polynomial([-nodes[other], 1]) / (nodes[state] - nodes[other])
    polynomial = (1 - leakage) * polynomial + Polynomial([leakage])
    a0, a1, a2 = np.pad(polynomial.coef, (0, 3 - len(polynomial.coef)))
    coefficients = [a0 + a2 / 2, a1, a2 / 2]
    walk = hamiltonian_walk(model)
    inner = lcu(coefficients, [np.eye(8), walk, walk @ walk])
    return Block(inner.circuit, inner.scale, 2)


def sequence_blocks(
    model: OpticalModel,
    m: int = 2,
    n: int = 1,
    leakage: float = 0.0,
) -> list[Block]:
    """Return execution sequence mu, P_n, mu, P_m, mu."""
    dipole = dipole_block(model)
    return [
        dipole,
        filter_block(model, n, leakage),
        dipole,
        filter_block(model, m, leakage),
        dipole,
    ]


def overlap_circuit(
    blocks: list[Block], quadrature: str | None = None
) -> QuantumCircuit:
    """Build coherent product with fresh work registers; ground is |00>.

    'real'/'imag' add a Hadamard-test control; None returns the product circuit.
    Qiskit little endian: system q[0:2], then work, then optional control.
    No intermediate measurement or renormalization occurs.
    """
    if quadrature not in (None, "real", "imag"):
        raise ValueError("Unknown quadrature")
    count = 2 + sum(block.ancillas for block in blocks)
    circuit = QuantumCircuit(count + (quadrature is not None))
    control = count
    if quadrature:
        circuit.h(control)
    cursor = 2
    for block in blocks:
        targets = [0, 1] + list(range(cursor, cursor + block.ancillas))
        matrix = block.matrix()
        if quadrature:
            controlled = np.eye(2 * len(matrix), dtype=complex)
            controlled[1::2, 1::2] = matrix
            circuit.append(
                UnitaryGate(controlled, label="controlled block"), [control] + targets
            )
        else:
            circuit.append(UnitaryGate(matrix, label="block"), targets)
        cursor += block.ancillas
    if quadrature:
        if quadrature == "imag":
            circuit.sdg(control)
        circuit.h(control)
    return circuit


def circuit_product(
    model: OpticalModel,
    m: int = 2,
    n: int = 1,
    leakage: float = 0.0,
) -> tuple[complex, float, QuantumCircuit]:
    """Return circuit overlap, normalization, circuit using exact statevector.

    The all-zero amplitude is the complex overlap probed by a Hadamard test.
    This exact simulator evaluation has no measurement noise.
    """
    blocks = sequence_blocks(model, m, n, leakage)
    circuit = overlap_circuit(blocks)
    scale = float(np.prod([block.scale for block in blocks]))
    state = Statevector.from_instruction(circuit)
    return complex(state.data[0] * scale), scale, circuit


def sample_hadamard(
    value: complex,
    scale: float,
    shots: int,
    rng: np.random.Generator,
) -> complex:
    """Classically sample Hadamard probabilities: shots PER quadrature.

    This is binomial sampling of circuit-derived probabilities, not hardware
    or a shot backend. Total cost is 2*shots circuit executions per product.
    """
    if shots <= 0 or scale <= 0:
        raise ValueError("Shots and normalization must be positive")
    output = []
    for component in (value.real, value.imag):
        probability = np.clip((1 + component / scale) / 2, 0, 1)
        output.append(scale * (2 * rng.binomial(shots, probability) / shots - 1))
    return complex(*output)
