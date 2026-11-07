"""Independent checks of Hamiltonian and optical-chirality diagnostics."""

import numpy as np
from numpy.testing import assert_allclose

from nii2_shg.hamiltonian_audit import (
    chirality_projection,
    fit_spin_hamiltonian,
    spin_operators,
)


def test_spin_fit_recovers_exchange_with_biquadratic_and_single_ion_terms():
    sp = spin_operators()
    rng = np.random.default_rng(12)
    exchange = np.array(
        [
            [-0.004, 0.0002, 0.0001],
            [0.0002, -0.0037, -0.00005],
            [0.0001, -0.00005, -0.003],
        ]
    )
    dot = sum(np.kron(s, s) for s in sp)
    b = -0.00089
    h = sum(
        exchange[i, j] * np.kron(sp[i], sp[j]) for i in range(3) for j in range(3)
    ) + b * (dot @ dot)
    q = sp[0] @ sp[0] - sp[2] @ sp[2]
    h += 0.0003 * np.kron(q, np.eye(3)) - 0.0001 * np.kron(np.eye(3), q)
    e, u = np.linalg.eigh(h)
    fit = fit_spin_hamiltonian(e, u.conj().T)
    assert_allclose(fit["exchange_tensor_mev"], exchange * 1000, atol=1e-12)
    assert_allclose(fit["biquadratic_mev"], b * 1000, atol=1e-12)
    assert fit["relative_matrix_residual"] < 1e-12
    # Arbitrary electronic eigenvector phases leave the spin-space Hamiltonian.
    phase = np.exp(1j * rng.uniform(-np.pi, np.pi, 9))
    fit2 = fit_spin_hamiltonian(e, phase[:, None] * u.conj().T)
    assert_allclose(fit2["exchange_tensor_mev"], fit["exchange_tensor_mev"], atol=1e-12)


def test_chirality_projection_recovers_three_components_in_any_low_basis():
    s = spin_operators()
    k = np.array(
        [
            np.kron(s[1], s[2]) - np.kron(s[2], s[1]),
            np.kron(s[2], s[0]) - np.kron(s[0], s[2]),
            np.kron(s[0], s[1]) - np.kron(s[1], s[0]),
        ]
    )
    rng = np.random.default_rng(8)
    c = rng.normal(size=(2, 2, 2, 3)) + 1j * rng.normal(size=(2, 2, 2, 3))
    r = np.linalg.qr(rng.normal(size=(9, 9)) + 1j * rng.normal(size=(9, 9)))[0]
    spin = np.einsum("ijka,apq->ijkpq", c, k)
    electronic = np.einsum("ap,ijkpq,bq->ijkab", r, spin, r.conj())
    diagnostic, projected = chirality_projection(electronic, r)
    assert_allclose(diagnostic["coefficients"], c, atol=1e-13)
    assert_allclose(projected, electronic, atol=1e-13)
    assert diagnostic["full_operator_relative_residual"] < 1e-13
