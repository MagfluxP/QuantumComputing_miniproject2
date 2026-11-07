"""Explicitly assumed quantum algorithm point-orbital SHG, not validated bulk NiI2.

Fan supplies electronic coefficients. A site-dependent Zeeman field, ideal
positions, length-gauge point charges and optical damping are added assumptions.
Classical response-space projection precedes the small-instance quantum circuit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, splu

from .fan_cluster import build_cluster, lift_one_body
from .fan_model import load_parameters


@dataclass(frozen=True)
class Assumptions:
    bond_length_angstrom: float = 2.8
    pinning_ev: float = 0.005
    pitch_deg: float = 49.68  # 360*0.138; a domain ansatz, not a fitted bond angle
    polarization_deg: float = 25.0  # local xy axes, not measured crystal axes
    gamma_ev: float = 0.5
    domain_sign: int = 1
    soc_sign: float = 1.0  # retain printed Eq.10; -1 is a convention sensitivity test


def operators(root: Path, assumptions: Assumptions) -> dict:
    """Construct H_Fan + H_pin and a physically dipole-odd point-charge operator.

    Ni_i=(0,0), Ni_j=(-l,l), I_A=(-l,0), I_B=(0,l). Orbital positions equal
    ionic centers; ionic static charge and an origin-dependent identity drop out
    of nonlinear response. Intra-orbital position/current matrix elements omitted.
    H_pin=-b*(n_i.S_i+n_j.S_j), n_i=x, n_j=cos(theta)x+sin(theta)y.
    This forces a local noncollinear state; it does not solve bulk spiral order.
    """
    if assumptions.bond_length_angstrom <= 0 or assumptions.gamma_ev <= 0:
        raise ValueError("Length and broadening must be positive")
    p = load_parameters(root)
    p["local"]["lambda_I"] *= assumptions.soc_sign
    c = build_cluster(p)
    theta = np.deg2rad(assumptions.pitch_deg)
    sx = np.array([[0, 1], [1, 0]], complex) / 2
    sy = np.array([[0, -1j], [1j, 0]], complex) / 2
    pin = np.zeros((20, 20), complex)
    pin[:4, :4] = -assumptions.pinning_ev * np.kron(np.eye(2), sx)
    pin[4:8, 4:8] = -assumptions.pinning_ev * np.kron(
        np.eye(2), np.cos(theta) * sx + np.sin(theta) * sy
    )
    if assumptions.domain_sign not in (-1, 1):
        raise ValueError("domain_sign must be +1 or -1")
    if assumptions.domain_sign == -1:
        pin[:4, :4], pin[4:8, 4:8] = pin[4:8, 4:8].copy(), pin[:4, :4].copy()
    h = c["h0_ev"] + lift_one_body(pin, c["basis"])
    site_positions = np.array([[0, 0], [-1, 1], [-1, 0], [0, 1]], float)
    site_positions -= [-0.5, 0.5]  # inversion center
    site_positions *= assumptions.bond_length_angstrom
    positions = np.repeat(site_positions, [4, 4, 6, 6], axis=0)
    phi = np.deg2rad(assumptions.polarization_deg)
    local_mu = -(positions @ [np.cos(phi), np.sin(phi)])
    mu = lift_one_body(np.diag(local_mu), c["basis"])
    return {"h": h, "mu": mu, "basis": c["basis"], "assumptions": asdict(assumptions)}


def response_basis(
    h: sparse.spmatrix,
    mu: sparse.spmatrix | Sequence[sparse.spmatrix],
    maximum: int = 128,
    photon_ev: float | None = None,
    gamma_ev: float = 0.5,
) -> tuple[np.ndarray, float, float]:
    """Build a classical response space with H/mu or shifted-inverse/mu actions.

    Ground state and optional frequency-targeted sparse factorizations are
    computed classically. The nested basis grows breadth-first with
    reorthogonalization. This is supporting reduction, not a quantum speedup.
    """
    start = perf_counter()
    rng = np.random.default_rng(51)
    energies, vectors = eigsh(
        h, k=1, which="SA", tol=1e-11, v0=rng.normal(size=h.shape[0])
    )
    g = vectors[:, 0]
    e0 = float(energies[0])
    cols = [g]
    dipoles = list(mu) if isinstance(mu, (list, tuple)) else [mu]
    dipole_actions = [lambda v, op=op: op @ v for op in dipoles]
    if photon_ev is None:
        actions = [lambda v: h @ v] + dipole_actions
    else:
        # Classical frequency-targeted rational reduction, not a quantum oracle.
        identity = sparse.eye(h.shape[0], format="csc")
        solves = [
            splu(
                (h - (e0 + sign * order * photon_ev + 1j * gamma_ev) * identity).tocsc()
            )
            for sign, order in [(1, 1), (-1, 1), (1, 2), (-1, 2)]
        ]
        actions = dipole_actions + [lu.solve for lu in solves]
    cursor = 0
    while len(cols) < maximum and cursor < len(cols):
        for action in actions:
            v = action(cols[cursor])
            # Double modified Gram-Schmidt prevents near-dependent Krylov vectors.
            q = np.column_stack(cols)
            for _ in range(2):
                v -= q @ (q.conj().T @ v)
            norm = np.linalg.norm(v)
            if norm > 1e-9:
                cols.append(v / norm)
            if len(cols) >= maximum:
                break
        cursor += 1
    return np.column_stack(cols), e0, perf_counter() - start


def project(
    h: np.ndarray | sparse.spmatrix,
    mu: np.ndarray | sparse.spmatrix,
    basis: np.ndarray,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return diagonal excitation energies and dipole in a nested response space."""
    q = basis[:, :size]
    hp = q.conj().T @ (h @ q)
    mp = q.conj().T @ (mu @ q)
    energies, rotation = np.linalg.eigh((hp + hp.conj().T) / 2)
    dipole = rotation.conj().T @ mp @ rotation
    # Remove the identity component; exact second-order response is invariant.
    dipole -= dipole[0, 0] * np.eye(len(energies))
    return energies - energies[0], (dipole + dipole.conj().T) / 2


def classical_response(
    energies: np.ndarray,
    mu: np.ndarray,
    photon_ev: float,
    gamma_ev: float,
    output_mu: np.ndarray | None = None,
    initial_rho: np.ndarray | None = None,
    second_gamma_factor: float = 1.0,
) -> complex:
    """Second-order recurrence for finite energies/dipoles and a stationary density.

    Inputs: energies (K,), Hermitian mu/output_mu (K,K), gamma_ev>0.
    initial_rho (K,K) defaults to the ground projector; returns complex beta.
    Hlm=-mu E; E=Eplus exp(-iwt)+c.c.; beta multiplies Eplus^2.
    Coefficient in (e Angstrom)^3/eV^2, not chi in pm/V.
    """
    if gamma_ev <= 0:
        raise ValueError("Broadening must be positive")
    rho0 = np.zeros_like(mu, dtype=complex)
    rho0[0, 0] = 1
    if initial_rho is not None:
        rho0 = np.asarray(initial_rho, dtype=complex)
    gaps = energies[:, None] - energies[None, :]
    rho1 = -(mu @ rho0 - rho0 @ mu) / (photon_ev - gaps + 1j * gamma_ev)
    rho2 = -(mu @ rho1 - rho1 @ mu) / (
        2 * photon_ev - gaps + 1j * gamma_ev * second_gamma_factor
    )
    return complex(np.trace((mu if output_mu is None else output_mu) @ rho2))
