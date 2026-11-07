"""Peierls-consistent point-orbital coupling and one unpinned spiral domain.

No ab-initio optical integrals are invented. H0 is Fan's finite nearest-bond
model; spin states are mapped through its isolated nine-state manifold. Optical
low-energy degeneracy and classical coherent-spin/phase averaging are explicit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, splu

from .fan_cluster import build_cluster, lift_one_body
from .fan_model import load_parameters, nearest_cluster_one_particle


@dataclass(frozen=True)
class SpiralConfig:
    length_angstrom: float = 2.8
    q_rlu: float = 0.138
    normal_tilt_deg: float = 55.0
    handedness: int = 1
    phase_samples: int = 8
    gamma_ev: float = 0.5
    second_gamma_factor: float = 2.0
    soc_sign: float = 1.0


DEFAULT_CONFIG = SpiralConfig()


C_AXIS = np.ones(3) / np.sqrt(3)
# a1=(-1,1,0)l, a2=(0,1,-1)l; q=(q,0) reciprocal-lattice units.
LAB_X = np.array([-2.0, 1, 1]) / np.sqrt(6)
LAB_Y = np.array([0.0, -1, 1]) / np.sqrt(2)
LAB = np.stack([LAB_X, LAB_Y, C_AXIS])
CYCLE = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)


def point_positions(length: float) -> np.ndarray:
    p = np.array([[0, 0, 0], [-1, 1, 0], [-1, 0, 0], [0, 1, 0]], float)
    p -= [-0.5, 0.5, 0]
    return np.repeat(p * length, [4, 4, 6, 6], axis=0)


def peierls_matrices(
    h: np.ndarray | sparse.spmatrix, mu: np.ndarray | sparse.spmatrix
) -> tuple[Any, Any]:
    """Jbar=hbar*d(mu)/dt and second A derivative for H(A)=e^(i mu A)H e^(-i mu A).

    H(A)=H - A*Jbar + A^2*D/2+...; charge and hbar are absorbed in A.
    Jbar units e Angstrom eV, D units (e Angstrom)^2 eV.
    """
    comm = h @ mu - mu @ h
    return 1j * comm, mu @ comm - comm @ mu


def cluster_inversion(basis: np.ndarray) -> sparse.csr_matrix:
    """Exact bond-center inversion with p-orbital and fermion permutation signs."""
    index = {int(state): i for i, state in enumerate(basis)}
    permutation = np.r_[
        np.arange(4, 8), np.arange(4), np.arange(14, 20), np.arange(8, 14)
    ]
    rows = []
    values = []
    for state in basis:
        occupied = [i for i in range(20) if int(state) & (1 << i)]
        transformed = [int(permutation[i]) for i in occupied]
        parity = sum(
            transformed[i] > transformed[j]
            for i in range(len(transformed))
            for j in range(i + 1, len(transformed))
        )
        parity += sum(i >= 8 for i in occupied)
        rows.append(index[sum(1 << i for i in transformed)])
        values.append((-1) ** parity)
    return sparse.csr_matrix(
        (values, (rows, np.arange(len(basis)))), shape=(len(basis), len(basis))
    )


def build_unpinned(root: Path, config: SpiralConfig = DEFAULT_CONFIG) -> dict[str, Any]:
    p = load_parameters(root)
    p["local"]["lambda_I"] *= config.soc_sign
    c = build_cluster(p)
    positions = point_positions(config.length_angstrom)
    local_dipoles = [
        lift_one_body(np.diag(-positions[:, i]), c["basis"]) for i in range(2)
    ]
    h1, _ = nearest_cluster_one_particle(p)
    one_dipoles = [np.diag(-positions[:, i]).astype(complex) for i in range(2)]
    inversion = cluster_inversion(c["basis"])
    residuals = {}
    inv_error = c["h0_ev"] @ inversion - inversion @ c["h0_ev"]
    if inv_error.nnz and np.max(abs(inv_error.data)) > 1e-10:
        raise ValueError("Hamiltonian inversion failed")
    for i, mu in enumerate(local_dipoles):
        j, d = peierls_matrices(c["h0_ev"], mu)
        j1, d1 = peierls_matrices(h1, one_dipoles[i])
        error = j - lift_one_body(j1, c["basis"])
        error2 = d - lift_one_body(d1, c["basis"])
        residuals["xy"[i]] = {
            "current_manybody_vs_onebody_max": float(np.max(abs(error.data)))
            if error.nnz
            else 0.0,
            "contact_manybody_vs_onebody_max": float(np.max(abs(error2.data)))
            if error2.nnz
            else 0.0,
        }
    return {
        **c,
        "dipoles": local_dipoles,
        "positions": positions,
        "h1p": h1,
        "one_dipoles": one_dipoles,
        "inversion": inversion,
        "operator_checks": residuals,
        "config": asdict(config),
    }


def nominal_spin_basis(basis: np.ndarray) -> np.ndarray:
    """Nine ionic |mi,mj> product triplets, mi,mj=+1,0,-1; ligands full."""
    index = {int(s): i for i, s in enumerate(basis)}
    triplet = [[(5, 1.0)], [(6, 1 / np.sqrt(2)), (9, 1 / np.sqrt(2))], [(10, 1.0)]]
    full_iodine = ((1 << 12) - 1) << 8
    out = np.zeros((len(basis), 9), complex)
    for col, (a, b) in enumerate(product(triplet, repeat=2)):
        for ai, av in a:
            for bi, bv in b:
                out[index[ai | (bi << 4) | full_iodine], col] = av * bv
    return out


def spin_mapping(ionic: np.ndarray, low: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Polar overlap gives an isometric, gauge-independent spin-to-electron map."""
    overlap = ionic.conj().T @ low
    u, s, vh = np.linalg.svd(overlap)
    if s.min() < 0.3:
        raise ValueError(f"Low manifold is not reliably spin-triplet-like: {s}")
    rotation = vh.conj().T @ u.conj().T
    return rotation, s


def low_manifold(h: sparse.spmatrix, basis: np.ndarray) -> dict[str, Any]:
    rng = np.random.default_rng(71)
    e, v = eigsh(
        h, k=12, which="SA", ncv=100, tol=1e-11, v0=rng.normal(size=h.shape[0])
    )
    order = np.argsort(e)
    e = e[order]
    v = v[:, order]
    # ARPACK's complex-Hermitian degenerate vectors need explicit orthogonalization.
    v = np.linalg.qr(v)[0]
    small = v.conj().T @ (h @ v)
    e, u = np.linalg.eigh((small + small.conj().T) / 2)
    v = v @ u
    low = v[:, :9]
    rotation, s = spin_mapping(nominal_spin_basis(basis), low)
    gap = float(e[9] - e[8])
    width = float(e[8] - e[0])
    if gap < 5 * width:
        raise ValueError("Nine-state manifold not spectrally isolated")
    return {
        "energies": e,
        "vectors": low,
        "spin_rotation": rotation,
        "overlap_singular_values": s,
        "gap_to_tenth_ev": gap,
        "low_width_ev": width,
        "eigen_residual_max": float(
            np.max(np.linalg.norm(h @ low - low * e[:9], axis=0))
        ),
    }


def shared_response_basis(
    h: sparse.spmatrix,
    dipoles: Sequence[sparse.spmatrix],
    low: dict[str, Any],
    maximum: int,
    photon_ev: float,
    gamma_ev: float,
    inversion: sparse.spmatrix | None = None,
) -> tuple[np.ndarray, float]:
    """Classical multi-start rational basis containing all nine spin states exactly."""
    start = perf_counter()
    e0 = float(np.mean(low["energies"][:9]))
    identity = sparse.eye(h.shape[0], format="csc")
    solves = [
        splu((h - (e0 + sign * order * photon_ev + 1j * gamma_ev) * identity).tocsc())
        for sign, order in [(1, 1), (-1, 1), (1, 2), (-1, 2)]
    ]
    actions = [lambda v, op=op: op @ v for op in dipoles] + [lu.solve for lu in solves]
    parity_actions = [(action, 0) for action in actions]
    if inversion is not None:
        # Every retained direction has definite parity. Reapply the projector
        # after cancellation during orthogonalization, before normalization.
        parity_actions = [(action, sign) for action in actions for sign in (1, -1)]
    q = np.empty((h.shape[0], maximum), complex)
    q[:, :9] = low["vectors"]
    count = 9
    cursor = 0
    while count < maximum and cursor < count:
        for action, parity in parity_actions:
            v = action(q[:, cursor])
            if parity:
                v = (v + parity * (inversion @ v)) / 2
            for _ in range(2):
                v -= q[:, :count] @ (q[:, :count].conj().T @ v)
                if parity:
                    v = (v + parity * (inversion @ v)) / 2
            norm = np.linalg.norm(v)
            if norm > 1e-9:
                q[:, count] = v / norm
                count += 1
            if count >= maximum:
                break
        cursor += 1
    return q[:, :count], perf_counter() - start


def project_unpinned(
    h: sparse.spmatrix,
    dipoles: Sequence[sparse.spmatrix],
    q: np.ndarray,
    size: int,
    ionic: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    """Project and flatten only the isolated low nine energies for frozen spins."""
    sub = q[:, :size]
    hp = sub.conj().T @ (h @ sub)
    e, u = np.linalg.eigh((hp + hp.conj().T) / 2)
    rotation, s = spin_mapping(ionic, sub @ u[:, :9])
    mean = float(np.mean(e[:9]))
    exc = e - mean
    exc[:9] = 0.0
    operators = []
    for mu in dipoles:
        m = u.conj().T @ (sub.conj().T @ (mu @ sub)) @ u
        m = (m + m.conj().T) / 2
        m -= np.trace(m[:9, :9]) / 9 * np.eye(size)
        operators.append(m)
    return exc, operators, rotation, s


def spin_coherent(direction: Sequence[float] | np.ndarray) -> np.ndarray:
    n = np.asarray(direction, float)
    n = n / np.linalg.norm(n)
    theta = np.arccos(np.clip(n[2], -1, 1))
    phi = np.arctan2(n[1], n[0])
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array(
        [np.exp(-1j * phi) * c * c, np.sqrt(2) * c * s, np.exp(1j * phi) * s * s]
    )


def domain_bonds(config: SpiralConfig = DEFAULT_CONFIG) -> list[dict[str, Any]]:
    """Three structural bonds in one domain; q phases follow reciprocal duality.

    Bond phases are [+2pi*q, 0, -2pi*q], not [delta,-delta/2,-delta/2].
    These are structural orientations, never three magnetic-domain weights.
    """
    l = config.length_angstrom
    a1 = l * np.array([-1.0, 1, 0])
    a2 = l * np.array([0.0, 1, -1])
    real_basis = np.column_stack([a1, a2, C_AXIS])
    reciprocal = 2 * np.pi * np.linalg.inv(real_basis).T
    wavevector = config.q_rlu * reciprocal[:, 0]
    result = []
    for name, power in [("Z", 0), ("X", 1), ("Y", 2)]:
        rotation = np.linalg.matrix_power(CYCLE, power)
        delta = rotation @ a1
        result.append(
            {
                "name": name,
                "rotation": rotation,
                "delta": delta,
                "phase_increment": float(wavevector @ delta),
                "optical_map": LAB[:2] @ rotation[:, :2],
            }
        )
    return result


def spiral_density(
    bond: dict[str, Any],
    config: SpiralConfig = DEFAULT_CONFIG,
    samples: int | None = None,
) -> np.ndarray:
    """Spatial phase-average within a fixed q/handedness/plane single domain."""
    count = config.phase_samples if samples is None else samples
    alpha = np.deg2rad(config.normal_tilt_deg)
    normal = np.sin(alpha) * LAB_X + np.cos(alpha) * C_AXIS
    u = LAB_Y
    v = config.handedness * np.cross(normal, u)
    rho = np.zeros((9, 9), complex)
    for phase in 2 * np.pi * np.arange(count) / count:
        ni = u * np.cos(phase) + v * np.sin(phase)
        nj = u * np.cos(phase + bond["phase_increment"]) + v * np.sin(
            phase + bond["phase_increment"]
        )
        state = np.kron(
            spin_coherent(bond["rotation"].T @ ni),
            spin_coherent(bond["rotation"].T @ nj),
        )
        rho += np.outer(state, state.conj()) / count
    return (rho + rho.conj().T) / 2


def effective_optical_tensor(
    energies: np.ndarray,
    dipoles: Sequence[np.ndarray],
    photon_ev: float,
    gamma_ev: float,
    second_gamma_factor: float = 1.0,
) -> np.ndarray:
    """Dual response operators O_ijk, before selecting any magnetic spin state.

    Tr(mu_out R2 C R1 C rho)=Tr(O rho). Transpose-dual resolvents have +gaps;
    two commutator-dual minus signs cancel. No optical tensor is fitted to data.
    """
    e = np.asarray(energies)
    gaps = e[:, None] - e[None, :]
    tensor = np.zeros((2, 2, 2, 9, 9), complex)
    for out in range(2):
        values = []
        b = dipoles[out] / (2 * photon_ev + gaps + 1j * gamma_ev * second_gamma_factor)
        for mu in [dipoles[0], dipoles[1], (dipoles[0] + dipoles[1]) / np.sqrt(2)]:
            c = mu @ b - b @ mu
            c = c / (photon_ev + gaps + 1j * gamma_ev)
            op = mu @ c - c @ mu
            values.append(op[:9, :9])
        tensor[out, 0, 0] = values[0]
        tensor[out, 1, 1] = values[1]
        tensor[out, 0, 1] = tensor[out, 1, 0] = values[2] - (values[0] + values[1]) / 2
    return tensor


def contract_domain(
    effective: np.ndarray,
    spin_rotation: np.ndarray,
    config: SpiralConfig = DEFAULT_CONFIG,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Coherently sum phase-averaged amplitudes over three structural bonds."""
    total = np.zeros((2, 2, 2), complex)
    details = []
    for bond in domain_bonds(config):
        rho_spin = spiral_density(bond, config)
        rho_low = spin_rotation @ rho_spin @ spin_rotation.conj().T
        local = np.einsum("ijkab,ba->ijk", effective, rho_low)
        a = bond["optical_map"]
        optical = np.einsum("ia,jb,kc,abc->ijk", a, a, a, local)
        total += optical
        details.append(
            {
                **bond,
                "rho_spin": rho_spin,
                "rho_low": rho_low,
                "local_tensor": local,
                "lab_tensor": optical,
            }
        )
    return total, details
