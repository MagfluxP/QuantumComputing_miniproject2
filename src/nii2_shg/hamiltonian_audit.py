"""Independent low-energy Hamiltonian and chirality diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np


def spin_operators() -> list[np.ndarray]:
    sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], complex) / np.sqrt(2)
    sy = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]]) / np.sqrt(2)
    return [sx, sy, np.diag([1.0, 0, -1])]


def fit_spin_hamiltonian(energies: np.ndarray, rotation: np.ndarray) -> dict[str, Any]:
    """Unique least-squares bond J tensor, B and two local quadrupole terms.

    H=constant+sum_ab J_ab Sia Sjb+B(Si.Sj)^2+local traceless quadratics.
    J_iso=trace(J)/3 is directly comparable across rotated coordinate systems;
    coefficients are fitted to electronic H, never to the SHG experimental data.
    """
    heff = (
        rotation.conj().T
        @ np.diag(np.asarray(energies)[:9] - np.mean(energies[:9]))
        @ rotation
    )
    sp = spin_operators()
    sx, sy, sz = sp
    identity = np.eye(3)
    operators = [np.eye(9)] + [np.kron(a, b) for a in sp for b in sp]
    dot = sum(np.kron(s, s) for s in sp)
    operators.append(dot @ dot)
    quadrupoles = [
        sx @ sx - sz @ sz,
        sy @ sy - sz @ sz,
        (sx @ sy + sy @ sx) / 2,
        (sx @ sz + sz @ sx) / 2,
        (sy @ sz + sz @ sy) / 2,
    ]
    operators += [
        np.kron(q, identity) if site == 0 else np.kron(identity, q)
        for site in [0, 1]
        for q in quadrupoles
    ]
    a = np.column_stack([op.reshape(-1) for op in operators])
    design = np.r_[a.real, a.imag]
    target = np.r_[heff.reshape(-1).real, heff.reshape(-1).imag]
    coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank != 21:
        raise ValueError("Spin Hamiltonian fit is not unique")
    exchange = coefficients[1:10].reshape(3, 3) * 1000
    return {
        "exchange_tensor_mev": exchange,
        "isotropic_exchange_mev": float(np.trace(exchange) / 3),
        "biquadratic_mev": float(coefficients[10] * 1000),
        "relative_matrix_residual": float(
            np.linalg.norm(design @ coefficients - target) / np.linalg.norm(target)
        ),
        "coefficient_order": "constant; Jxx,Jxy,Jxz,Jyx,Jyy,Jyz,Jzx,Jzy,Jzz; B; five traceless quadrupoles on site i, then j",
        "coefficients_ev": coefficients,
    }


def chirality_projection(
    effective: np.ndarray, rotation: np.ndarray
) -> tuple[dict[str, Any], np.ndarray]:
    """Hilbert-Schmidt projection on (Si cross Sj), without changing the model."""
    spins = spin_operators()
    chirality = np.array(
        [
            np.kron(spins[1], spins[2]) - np.kron(spins[2], spins[1]),
            np.kron(spins[2], spins[0]) - np.kron(spins[0], spins[2]),
            np.kron(spins[0], spins[1]) - np.kron(spins[1], spins[0]),
        ]
    )
    spin_operator = np.einsum(
        "ap,ijkab,bq->ijkpq", rotation.conj(), effective, rotation
    )
    coefficients = np.einsum(
        "apq,ijkpq->ijka", chirality.conj(), spin_operator
    ) / np.sum(abs(chirality) ** 2, axis=(1, 2))
    projected_spin = np.einsum("ijka,apq->ijkpq", coefficients, chirality)
    projected_electronic = np.einsum(
        "ap,ijkpq,bq->ijkab", rotation, projected_spin, rotation.conj()
    )
    norms = np.sqrt(np.sum(abs(coefficients) ** 2, axis=(0, 1, 2)))
    return {
        "coefficients": coefficients,
        "component_norms_xyz": norms,
        "inplane_over_normal_coefficient_norm": float(
            np.linalg.norm(norms[:2]) / max(norms[2], 1e-30)
        ),
        "full_operator_relative_residual": float(
            np.linalg.norm(spin_operator - projected_spin)
            / np.linalg.norm(spin_operator)
        ),
    }, projected_electronic
