"""Physics and encoding checks for the Gao spin-H0 branch."""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm

from nii2_shg.gao_spin_model import (
    GaoSpinConfig,
    MissingGaoOpticalOperator,
    build_gao_h0,
    embed_qutrit,
    encode_local_hamiltonian,
    encoded_indices,
    exchange_tensor,
    geometry,
    require_optical_shg_inputs,
)
from nii2_shg.hamiltonian_audit import spin_operators

ROOT = Path(__file__).resolve().parents[1]


def test_source_shells_and_exchange_trace():
    _, _, bonds = geometry()
    assert [sum(b["shell"] == s for b in bonds) for s in [1, 2, 3]] == [12, 6, 3]
    p = json.loads((ROOT / "data/nii2/gao2024_parameters.json").read_text())
    for shell, expected in [(1, -5.03), (2, -0.32), (3, 3.95)]:
        tensor = exchange_tensor(p, shell, 0)
        assert_allclose(np.trace(tensor) / 3, expected * 1e-3, atol=1e-15)
        assert_allclose(
            exchange_tensor(p, shell, 2 * np.pi / 3),
            exchange_tensor(p, shell, 5 * np.pi / 3),
            atol=1e-15,
        )
    assert_allclose(
        exchange_tensor(p, 2, 0, GaoSpinConfig("main_text")), 0.32e-3 * np.eye(3)
    )


def test_qutrit_encoding_preserves_exact_local_evolution_without_leakage():
    sp = spin_operators()
    h = sum(np.kron(s, s) for s in sp) * 0.003
    h += 0.0004 * np.kron(sp[0], sp[2])
    encoded = encode_local_hamiltonian(h)
    indices = encoded_indices(2)
    unitary = expm(-1j * encoded * 37)
    assert_allclose(unitary[np.ix_(indices, indices)], expm(-1j * h * 37), atol=1e-14)
    unused = np.setdiff1d(np.arange(16), indices)
    assert_allclose(unitary[np.ix_(unused, indices)], 0, atol=1e-14)


def test_embedding_site_order_and_spin_quantization():
    sx, sy, sz = spin_operators()
    assert_allclose(sx @ sy - sy @ sx, 1j * sz, atol=1e-15)
    assert_allclose(embed_qutrit(sz, [0], 2).toarray(), np.kron(np.eye(3), sz))
    assert_allclose(embed_qutrit(sz, [1], 2).toarray(), np.kron(sz, np.eye(3)))
    local = np.kron(sy, sx)
    assert_allclose(embed_qutrit(local, [0, 1], 2).toarray(), local)


def test_full_cluster_is_hermitian_and_product_energy_matches_local_terms():
    c = build_gao_h0(ROOT)
    h = c["h0_ev"]
    delta = h - h.conj().T
    assert delta.nnz == 0 or np.max(abs(delta.data)) < 1e-14
    rng = np.random.default_rng(81)
    states = []
    for _ in range(7):
        v = rng.normal(size=3) + 1j * rng.normal(size=3)
        states.append(v / np.linalg.norm(v))
    psi = np.array([1.0 + 0j])
    for state in reversed(states):
        psi = np.kron(psi, state)
    expected = 0j
    for term in c["terms"]:
        local = np.array([1.0 + 0j])
        for site in reversed(term["sites"]):
            local = np.kron(local, states[site])
        expected += np.vdot(local, term["matrix_ev"] @ local)
    assert_allclose(np.vdot(psi, h @ psi), expected, atol=1e-14)


def test_optical_shg_cannot_be_inferred_from_spin_h0_alone():
    with pytest.raises(MissingGaoOpticalOperator, match="matched frequency-dependent"):
        require_optical_shg_inputs()
