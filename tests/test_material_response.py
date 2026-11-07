"""Check coherent response against independent density matrices and symmetries."""

from pathlib import Path

import numpy as np

from nii2_shg.material_response import (
    Assumptions,
    classical_response,
    operators,
    project,
)
from nii2_shg.response_circuit import dilation, quantum_response

ROOT = Path(__file__).resolve().parents[1]


def test_unitary_dilation_nonhermitian():
    rng = np.random.default_rng(5)
    a = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    u, scale = dilation(a)
    np.testing.assert_allclose(u.conj().T @ u, np.eye(8), atol=3e-8)
    np.testing.assert_allclose(scale * u[:4, :4], a, atol=1e-12)


def test_quantum_response_full_commutator_and_readout():
    rng = np.random.default_rng(19)
    m = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    m = (m + m.conj().T) / 4  # include complex transitions and permanent dipoles
    e = np.array([0.0, 1.1, 2.6, 4.0])
    expected = classical_response(e, m, 1.20373, 0.5)
    q = quantum_response(e, m, 1.20373, 0.5)
    np.testing.assert_allclose(q["beta"], expected, atol=2e-11)
    np.testing.assert_allclose(q["hadamard_beta"], expected, atol=2e-10)
    np.testing.assert_allclose(
        classical_response(e, m + 2 * np.eye(4), 1.20373, 0.5), expected, atol=1e-12
    )


def test_point_orbital_and_domain_inversion():
    a = operators(ROOT, Assumptions())
    b = operators(ROOT, Assumptions(domain_sign=-1))
    permutation = (
        list(range(4, 8)) + list(range(4)) + list(range(14, 20)) + list(range(8, 14))
    )
    rows = []
    signs = []
    index = {int(s): i for i, s in enumerate(a["basis"])}
    for state in a["basis"]:
        occupied = [p for p in range(20) if int(state) & (1 << p)]
        target = [permutation[p] for p in occupied]
        swaps = sum(x > y for i, x in enumerate(target) for y in target[i + 1 :])
        signs.append((-1) ** (swaps + sum(p >= 8 for p in occupied)))
        rows.append(index[sum(1 << p for p in target)])
    from scipy.sparse import coo_matrix

    inversion = coo_matrix(
        (signs, (rows, range(len(rows)))), shape=a["h"].shape
    ).tocsr()
    for residual in [
        inversion @ a["h"] @ inversion.T - b["h"],
        inversion @ a["mu"] @ inversion.T + a["mu"],
    ]:
        assert not residual.nnz or np.max(abs(residual.data)) < 1e-12


def test_reduction_preserves_ground_and_dipole_parity_zero():
    # A complete finite model with a non-diagonal basis checks projection conventions.
    rng = np.random.default_rng(11)
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    e = np.array([0.0, 1.0, 2.1, 3.2])
    mu = np.array(
        [[0, 1, 0, 0.1], [1, 0, 0.5, 0], [0, 0.5, 0, 0.3], [0.1, 0, 0.3, 0]], complex
    )
    h = np.diag(e)
    ep, mp = project(h, mu, q, 4)
    np.testing.assert_allclose(ep, e, atol=1e-12)
    assert abs(classical_response(ep, mp, 1.2, 0.5)) < 1e-12
