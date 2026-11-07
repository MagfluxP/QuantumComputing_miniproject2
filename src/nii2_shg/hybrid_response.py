"""Gao magnetic pair states contracted with a fixed electronic SHG operator.

No Gao spin energies replace optical electronic denominators. The optional
chirality source selects one domain in a finite cluster and is not a published
Gao coupling. Coordinates follow the existing explicit table-caption choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from scipy.linalg import expm
from scipy.sparse.linalg import eigsh
from scipy.spatial.transform import Rotation

from .consistent_spiral import LAB, domain_bonds
from .gao_spin_model import embed_qutrit
from .hamiltonian_audit import spin_operators

KB_EV_K = 8.617333262145e-5


@dataclass(frozen=True)
class HybridStateConfig:
    temperature_k: float = 1.63
    selector_mev: float = 0.05
    q_rlu: float = 0.138
    normal_tilt_deg: float = 55.0
    handedness: int = 1
    eigenstates: int = 24


DEFAULT_CONFIG = HybridStateConfig()


def chirality_operators() -> np.ndarray:
    s = spin_operators()
    # Local site order is i (least-significant) then j for embed_qutrit.
    return np.array(
        [
            np.kron(s[2], s[1]) - np.kron(s[1], s[2]),
            np.kron(s[0], s[2]) - np.kron(s[2], s[0]),
            np.kron(s[1], s[0]) - np.kron(s[0], s[1]),
        ]
    )


def domain_selector(
    cluster: dict[str, Any], config: HybridStateConfig = DEFAULT_CONFIG
) -> sparse.csr_matrix:
    """-h sum_NN sin(q.delta) n.(Si cross Sj), time-reversal even.

    Lab x is along q. With a1=(0,1),a2=(sqrt(3)/2,1/2), q=q_rlu*b2.
    This is the same physical in-plane propagation as the optical branch, in
    a different explicitly stated primitive-vector convention.
    """
    normal = np.array(
        [
            np.sin(np.deg2rad(config.normal_tilt_deg)),
            0,
            np.cos(np.deg2rad(config.normal_tilt_deg)),
        ]
    )
    wavevector = np.array([4 * np.pi * config.q_rlu / np.sqrt(3), 0.0])
    local = np.einsum("a,aij->ij", normal, chirality_operators())
    selector = sparse.csr_matrix(cluster["h0_ev"].shape, dtype=complex)
    for bond in cluster["bonds"]:
        if bond["shell"] != 1:
            continue
        i, j = bond["i"], bond["j"]
        phase = wavevector @ (cluster["positions"][j] - cluster["positions"][i])
        selector += (
            -config.selector_mev
            * 1e-3
            * config.handedness
            * np.sin(phase)
            * embed_qutrit(local, [i, j], 7)
        )
    return selector


def selected_state(
    cluster: dict[str, Any], config: HybridStateConfig = DEFAULT_CONFIG
) -> dict[str, Any]:
    """Return low eigenpairs, thermal weights and tail bound for Gao H0+selector. Inputs: cluster metadata and nonnegative T/source config; spin diagonalization is classical."""
    if config.temperature_k < 0 or config.selector_mev < 0:
        raise ValueError("Temperature and selector magnitude must be nonnegative")
    h = cluster["h0_ev"] + domain_selector(cluster, config)
    e, v = eigsh(
        h,
        k=config.eigenstates,
        which="SA",
        ncv=max(90, 3 * config.eigenstates),
        tol=1e-11,
        v0=np.random.default_rng(915).normal(size=h.shape[0]),
    )
    order = np.argsort(e)
    v = np.linalg.qr(v[:, order])[0]
    small = v.conj().T @ (h @ v)
    e, u = np.linalg.eigh((small + small.conj().T) / 2)
    v = v @ u
    if config.temperature_k == 0:
        weights = (e - e[0] < 1e-9).astype(float)
        weights /= weights.sum()
        tail = 0.0
    else:
        boltz = np.exp(-(e - e[0]) / (KB_EV_K * config.temperature_k))
        weights = boltz / boltz.sum()
        tail = float(
            (h.shape[0] - len(e))
            * np.exp(-(e[-1] - e[0]) / (KB_EV_K * config.temperature_k))
            / boltz.sum()
        )
        if tail > 1e-7:
            raise ValueError("Too few spin eigenstates for a controlled thermal trace")
    return {
        "energies": e,
        "vectors": v,
        "weights": weights,
        "thermal_tail_bound": tail,
        "eigen_residual_max_ev": float(np.max(np.linalg.norm(h @ v - v * e, axis=0))),
        "config": config,
    }


def pair_density(
    vectors: np.ndarray, weights: np.ndarray, i: int, j: int, count: int = 7
) -> np.ndarray:
    """Pair matrix is big-endian |mi,mj>, as the optical product-spin basis."""
    axes = [count - 1 - i, count - 1 - j]
    rest = [k for k in range(count) if k not in axes]
    rho = np.zeros((9, 9), complex)
    for state, weight in zip(vectors.T, weights, strict=True):
        if weight < 1e-16:
            continue
        pair = state.reshape((3,) * count).transpose(axes + rest).reshape(9, -1)
        rho += weight * (pair @ pair.conj().T)
    return (rho + rho.conj().T) / 2


def spin_rotation(matrix: np.ndarray) -> np.ndarray:
    """Unitary for an active Cartesian rotation, checked independently in tests."""
    vector = Rotation.from_matrix(matrix).as_rotvec()
    return expm(-1j * sum(vector[a] * spin_operators()[a] for a in range(3)))


def oriented_pair_states(
    cluster: dict[str, Any], state: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Four NN pair densities per orientation, all from one selected domain."""
    canonical = domain_bonds()
    directions = np.array(
        [LAB[:2] @ b["delta"] / np.linalg.norm(b["delta"]) for b in canonical]
    )
    results = {
        b["name"]: {
            "density": np.zeros((9, 9), complex),
            "count": 0,
            "optical_map": b["optical_map"],
            "bonds": [],
        }
        for b in canonical
    }
    swap = np.eye(9).reshape(3, 3, 3, 3).transpose(1, 0, 2, 3).reshape(9, 9)
    for bond in cluster["bonds"]:
        if bond["shell"] != 1:
            continue
        i, j = bond["i"], bond["j"]
        delta = cluster["positions"][j] - cluster["positions"][i]
        delta /= np.linalg.norm(delta)
        dots = directions @ delta
        index = int(np.argmax(abs(dots)))
        b = canonical[index]
        if abs(dots[index]) < 1 - 1e-10:
            raise ValueError("Magnetic and optical bond geometries do not align")
        rho = pair_density(state["vectors"], state["weights"], i, j)
        if dots[index] < 0:
            rho = swap @ rho @ swap.conj().T
        d = spin_rotation(b["rotation"].T @ LAB.T)
        pair_rotation = np.kron(d, d)
        rho = pair_rotation @ rho @ pair_rotation.conj().T
        result = results[b["name"]]
        result["density"] += rho
        result["count"] += 1
        result["bonds"].append([i, j])
    for result in results.values():
        result["density"] /= result["count"]
    return results


def hybrid_tensor(
    effective: np.ndarray, rotation: np.ndarray, oriented: dict[str, dict[str, Any]]
) -> np.ndarray:
    """Three orientation-averaged complex amplitudes, summed before squaring."""
    total = np.zeros((2, 2, 2), complex)
    for result in oriented.values():
        rho = rotation @ result["density"] @ rotation.conj().T
        local = np.einsum("ijkab,ba->ijk", effective, rho)
        a = result["optical_map"]
        total += np.einsum("ia,jb,kc,abc->ijk", a, a, a, local)
    return total
