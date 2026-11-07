"""Phase-sequence QSP for a smoothed spectral indicator on a small Hamiltonian.

Implements the construction, not a built-in QSP solver. Phase synthesis uses
SciPy least squares and is numerically validated, not a general synthesis proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.polynomial import Chebyshev
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator
from scipy.optimize import least_squares
from scipy.special import erf

from .circuits import Block, dipole_block, lcu, overlap_circuit
from .model import OpticalModel


def window(x: np.ndarray, halfwidth: float = 0.22, sharpness: float = 18) -> np.ndarray:
    """Even erf-smoothed indicator; endpoints have a finite transition region."""
    return 0.5 * (erf(sharpness * (x + halfwidth)) - erf(sharpness * (x - halfwidth)))


def scalar_qsp(x: np.ndarray, phases: np.ndarray) -> np.ndarray:
    """Evaluate Re[U00(x)] for R(phi0) W(x) R(phi1) ... W(x) R(phid)."""
    x = np.asarray(x, dtype=float).reshape(-1)
    root = np.sqrt(np.clip(1 - x**2, 0, None))
    signal = np.zeros((len(x), 2, 2), dtype=complex)
    signal[:, 0, 0] = signal[:, 1, 1] = x
    signal[:, 0, 1], signal[:, 1, 0] = root, -root
    unitary = np.broadcast_to(np.eye(2, dtype=complex), signal.shape).copy()
    for j, phase in enumerate(phases):
        unitary = unitary @ np.diag([np.exp(1j * phase), np.exp(-1j * phase)])
        if j < len(phases) - 1:
            unitary = unitary @ signal
    return unitary[:, 0, 0].real


@dataclass(frozen=True)
class QSPDesign:
    """Classical design outputs; errors are measured over validation grids."""

    polynomial: Chebyshev
    phases: np.ndarray
    synthesis_error: float
    window_error: float
    degree: int


@lru_cache(maxsize=24)
def design_window(
    degree: int = 32, halfwidth: float = 0.22, sharpness: float = 18
) -> QSPDesign:
    """Fit a bounded even polynomial and solve its phase sequence reproducibly.

    Polynomial extrema bound the target over [-1,1]. A separate dense grid checks
    phase synthesis; failures raise rather than silently substituting a projector.
    """
    if (
        degree < 2
        or degree > 64
        or degree % 2
        or not 0 < halfwidth < 1
        or sharpness <= 0
    ):
        raise ValueError("Use even degree 2..64, 0<halfwidth<1, sharpness>0")
    polynomial = Chebyshev.interpolate(
        lambda x: window(x, halfwidth, sharpness), degree
    )
    coefficients = polynomial.coef.copy()
    coefficients[1::2] = 0
    polynomial = Chebyshev(coefficients)
    roots = polynomial.deriv().roots()
    extrema = [float(r.real) for r in roots if abs(r.imag) < 1e-10 and abs(r.real) <= 1]
    maximum = np.max(abs(polynomial(np.array([-1, 1] + extrema))))
    polynomial = polynomial / (max(1.0, maximum) * (1 + 1e-7))
    nodes = np.cos(np.pi * (np.arange(4 * degree + 8) + 0.5) / (4 * degree + 8))
    grid = np.linspace(-1, 1, 4001)
    best = None
    for seed in range(4):
        initial = np.random.default_rng(seed).normal(0, 0.15, degree + 1)
        initial[[0, -1]] += np.pi / 4
        fit = least_squares(
            lambda phases: scalar_qsp(nodes, phases) - polynomial(nodes),
            initial,
            max_nfev=600,
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-11,
        )
        error = float(np.max(abs(scalar_qsp(grid, fit.x) - polynomial(grid))))
        if best is None or error < best[0]:
            best = error, fit.x
        if error < 1e-8:
            break
    if best[0] > 1e-6:
        raise RuntimeError(f"QSP synthesis failed validation: {best[0]}")
    return QSPDesign(
        polynomial,
        best[1],
        best[0],
        float(np.max(abs(polynomial(grid) - window(grid, halfwidth, sharpness)))),
        degree,
    )


def qsp_signal_circuit(
    model: OpticalModel, center_ev: float, degree: int = 32
) -> tuple[QuantumCircuit, QSPDesign]:
    """Build repeated signal gates and ancilla phase rotations for the filter.

    Center and width are supplied window choices, not computed eigenvectors.
    A norm bound rescales H-center to [-1,1]. A small diagonal square root builds the signal
    oracle; it is supporting preprocessing, not a scalable chemistry oracle.
    """
    design = design_window(degree)
    shifted = model.hamiltonian - center_ev * np.eye(4)
    radius = max(float(np.linalg.norm(shifted, ord=np.inf)), 1e-12)
    h = shifted / radius
    # Supporting 4x4 oracle construction; diagonal toy H requires no eigenvectors.
    if not np.allclose(h, np.diag(np.diag(h))):
        raise ValueError("This teaching oracle supports diagonal optical models only")
    root = np.diag(np.sqrt(np.clip(1 - np.diag(h).real ** 2, 0, None)))
    signal = np.block([[h, root], [-root, h]])
    circuit = QuantumCircuit(3)
    # Append reverse order to match the rightmost-first matrix convention.
    for j in range(degree, -1, -1):
        circuit.rz(-2 * float(design.phases[j]), 2)
        if j > 0:
            circuit.append(UnitaryGate(signal, label="signal(H-center)"), range(3))
    return circuit, design


def qsp_filter(
    model: OpticalModel, center_ev: float, degree: int = 32
) -> tuple[Block, QSPDesign]:
    """Select the real polynomial from the explicitly constructed QSP sequence."""
    circuit, design = qsp_signal_circuit(model, center_ev, degree)
    unitary = np.asarray(Operator(circuit).data)
    # Hermitian part selects the real polynomial, retaining its sign and phase.
    real_part = lcu([0.5, 0.5], [unitary, unitary.conj().T])
    return Block(real_part.circuit, real_part.scale, 2), design


def qsp_sequence(
    model: OpticalModel, m: int = 2, n: int = 1, degree: int = 32
) -> list[Block]:
    """Return coherent dipole/filter sequence; ground preparation is |00>."""
    if m not in (0, 1, 2) or n not in (0, 1, 2):
        raise ValueError("Only active-state windows 0,1,2 are supported")
    dipole = dipole_block(model)
    first, _ = qsp_filter(model, float(model.energies[n]), degree)
    second, _ = qsp_filter(model, float(model.energies[m]), degree)
    return [dipole, first, dipole, second, dipole]


def qsp_product(
    model: OpticalModel, m: int = 2, n: int = 1, degree: int = 32
) -> tuple[complex, float, QuantumCircuit]:
    """Return norm-restored complex product, block scale and coherent circuit."""
    from qiskit.quantum_info import Statevector

    blocks = qsp_sequence(model, m, n, degree)
    circuit = overlap_circuit(blocks)
    scale = float(np.prod([b.scale for b in blocks]))
    result = Statevector.from_instruction(circuit).data[0] * scale
    return complex(result), scale, circuit
