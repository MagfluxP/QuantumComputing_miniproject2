"""Frame, pair-order and domain-selection checks independent of SHG fitting."""

from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
from scipy import sparse
from scipy.spatial.transform import Rotation

from nii2_shg.gao_spin_model import build_gao_h0
from nii2_shg.hamiltonian_audit import spin_operators
from nii2_shg.hybrid_response import (
    HybridStateConfig,
    domain_selector,
    pair_density,
    spin_rotation,
)

ROOT = Path(__file__).resolve().parents[1]


def test_partial_trace_preserves_pair_order_in_little_endian_full_state():
    rng = np.random.default_rng(47)
    states = []
    for _ in range(4):
        v = rng.normal(size=3) + 1j * rng.normal(size=3)
        states.append(v / np.linalg.norm(v))
    full = np.array([1.0 + 0j])
    for v in reversed(states):
        full = np.kron(full, v)
    rho = pair_density(full[:, None], np.ones(1), 0, 3, count=4)
    expected = np.kron(states[0], states[3])
    assert_allclose(rho, np.outer(expected, expected.conj()), atol=1e-14)
    # Entangled pair has a genuinely mixed one-site reduction, not product fitting.
    bell = np.eye(3).reshape(-1) / np.sqrt(3)
    pair = pair_density(bell[:, None], np.ones(1), 0, 1, count=2)
    assert_allclose(
        pair.reshape(3, 3, 3, 3).trace(axis1=1, axis2=3), np.eye(3) / 3, atol=1e-14
    )


def test_spin_frame_rotation_matches_cartesian_vector_rotation():
    rng = np.random.default_rng(94)
    v = rng.normal(size=3) + 1j * rng.normal(size=3)
    v /= np.linalg.norm(v)
    matrix = Rotation.from_rotvec([0.8, -0.6, 0.3]).as_matrix()
    d = spin_rotation(matrix)
    s = spin_operators()
    before = np.array([np.vdot(v, a @ v).real for a in s])
    after = np.array([np.vdot(d @ v, a @ d @ v).real for a in s])
    assert_allclose(after, matrix @ before, atol=1e-14)
    assert_allclose(d.conj().T @ d, np.eye(3), atol=1e-14)


def test_domain_selector_is_odd_under_cluster_inversion():
    c = build_gao_h0(ROOT)
    positions = c["positions"]
    site_map = [
        int(np.argmin(np.linalg.norm(positions + p, axis=1))) for p in positions
    ]
    dimension = 3**7
    rows = []
    for col in range(dimension):
        rows.append(sum((col // 3**i % 3) * 3 ** site_map[i] for i in range(7)))
    inversion = sparse.csr_matrix(
        (np.ones(dimension), (rows, np.arange(dimension))), shape=(dimension, dimension)
    )
    field = domain_selector(c)
    error = inversion @ field @ inversion + field
    assert error.nnz == 0 or np.max(abs(error.data)) < 1e-14
    herror = inversion @ c["h0_ev"] @ inversion - c["h0_ev"]
    assert herror.nnz == 0 or np.max(abs(herror.data)) < 1e-14
    assert domain_selector(c, HybridStateConfig(selector_mev=0)).nnz == 0


def test_chirality_source_is_time_reversal_even():
    from scipy.linalg import expm

    from nii2_shg.hybrid_response import chirality_operators

    s = spin_operators()
    flip = expm(-1j * np.pi * s[1])
    pair = np.kron(flip, flip)
    for op in chirality_operators():
        assert_allclose(pair @ op.conj() @ pair.conj().T, op, atol=1e-14)
