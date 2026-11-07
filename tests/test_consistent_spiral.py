"""Physical consistency checks independent of the experimental fit."""

import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import expm

from nii2_shg.angular_response import classical_tensor
from nii2_shg.consistent_spiral import (
    SpiralConfig,
    domain_bonds,
    effective_optical_tensor,
    peierls_matrices,
    spin_coherent,
    spin_mapping,
    spiral_density,
)
from nii2_shg.material_response import classical_response
from nii2_shg.response_circuit import quantum_response


def test_peierls_derivatives_and_length_velocity_gauge():
    rng = np.random.default_rng(8)
    e = np.arange(4.0) * 0.7
    h = np.diag(e)
    a = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    mu = (a + a.conj().T) / 3
    j, d = peierls_matrices(h, mu)
    step = 1e-4
    hp = expm(1j * mu * step) @ h @ expm(-1j * mu * step)
    hm = expm(-1j * mu * step) @ h @ expm(1j * mu * step)
    assert_allclose((hp - hm) / (2 * step), -j, atol=1e-7)
    assert_allclose((hp + hm - 2 * h) / step**2, d, atol=2e-7)
    rho = np.diag([1.0, 0, 0, 0])
    w = 1.2
    g = 0.13
    z = w + 1j * g
    gaps = e[:, None] - e[None, :]
    v = 1j * j / z
    v2 = -d / (2 * z * z)
    r1 = (v @ rho - rho @ v) / (z - gaps)
    r2 = (v @ r1 - r1 @ v + v2 @ rho - rho @ v2) / (2 * z - gaps)
    length = classical_response(e, mu, w, g, second_gamma_factor=2.0)
    assert_allclose(np.trace(mu @ r2), length, atol=1e-11)
    without_contact = np.trace(mu @ ((v @ r1 - r1 @ v) / (2 * z - gaps)))
    assert abs(without_contact - length) > 1e-3


def test_one_spiral_reciprocal_geometry_and_phase_average():
    bonds = domain_bonds()
    delta = 2 * np.pi * 0.138
    assert_allclose(
        [b["phase_increment"] for b in bonds], [delta, 0, -delta], atol=1e-15
    )
    for b in bonds:
        rho = spiral_density(b)
        assert_allclose(np.trace(rho), 1, atol=1e-14)
        assert np.linalg.eigvalsh(rho).min() > -1e-13
        assert_allclose(rho, spiral_density(b, samples=16), atol=1e-14)
        assert_allclose(rho, spiral_density(b, samples=32), atol=1e-14)
    # Coherent S=1 spin points along its declared direction.
    sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2)
    sy = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]]) / np.sqrt(2)
    sz = np.diag([1.0, 0, -1])
    n = np.array([0.3, -0.4, np.sqrt(0.75)])
    state = spin_coherent(n)
    assert_allclose(
        [np.vdot(state, s @ state).real for s in [sx, sy, sz]], n, atol=1e-14
    )
    # q=0 gives identical spin pairs on every bond, with no helical chirality.
    for b in domain_bonds(SpiralConfig(q_rlu=0)):
        rho = spiral_density(b, SpiralConfig(q_rlu=0))
        swapped = rho.reshape(3, 3, 3, 3).transpose(1, 0, 3, 2).reshape(9, 9)
        assert_allclose(rho, swapped, atol=1e-14)


def test_polar_spin_map_is_gauge_independent():
    rng = np.random.default_rng(3)
    low = np.eye(12, dtype=complex)[:, :9]
    ionic = low.copy()
    a = rng.normal(size=(9, 9)) + 1j * rng.normal(size=(9, 9))
    u = np.linalg.qr(a)[0]
    r, s = spin_mapping(ionic, low @ u)
    assert_allclose(low @ u @ r, ionic, atol=1e-14)
    assert_allclose(s, 1, atol=1e-14)


def test_dual_optical_tensor_matches_density_response():
    rng = np.random.default_rng(15)
    e = np.r_[np.zeros(9), np.linspace(1, 4, 7)]
    dipoles = []
    for _ in range(2):
        a = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
        dipoles.append((a + a.conj().T) / 20)
    a = rng.normal(size=(9, 3)) + 1j * rng.normal(size=(9, 3))
    rho9 = a @ a.conj().T
    rho9 /= np.trace(rho9)
    rho = np.zeros((16, 16), complex)
    rho[:9, :9] = rho9
    operators = effective_optical_tensor(e, dipoles, 1.2, 0.5, 2.0)
    expected = classical_tensor(e, dipoles, 1.2, 0.5, rho, 2.0)
    assert_allclose(np.einsum("ijkab,ba->ijk", operators, rho9), expected, atol=1e-13)


def test_density_preparation_circuit_and_hadamard():
    e = np.array([0.0, 0.0, 1.1, 2.4])
    rng = np.random.default_rng(9)
    a = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    mu = (a + a.conj().T) / 3
    rho = np.zeros((4, 4), complex)
    rho[:2, :2] = [[0.6, 0.1j], [-0.1j, 0.4]]
    expected = classical_response(
        e, mu, 1.2, 0.2, initial_rho=rho, second_gamma_factor=2.0
    )
    result = quantum_response(e, mu, 1.2, 0.2, initial_rho=rho, second_gamma_factor=2.0)
    assert_allclose(result["beta"], expected, atol=2e-10)
    assert_allclose(result["hadamard_beta"], expected, atol=2e-10)


def test_dense_preparation_handles_nearly_zero_amplitudes():
    from nii2_shg.response_circuit import state_preparation_unitary

    rng = np.random.default_rng(62)
    v = np.zeros(256, complex)
    v[12:19] = rng.normal(size=7) + 1j * rng.normal(size=7)
    v[0] = 1e-18j
    v /= np.linalg.norm(v)
    u = state_preparation_unitary(v)
    assert_allclose(u[:, 0], v, atol=2e-15)
    assert_allclose(u.conj().T @ u, np.eye(256), atol=2e-15)


def test_exact_cluster_inversion_and_dipole_parity():
    from nii2_shg.consistent_spiral import (
        cluster_inversion,
        nominal_spin_basis,
        point_positions,
    )
    from nii2_shg.fan_cluster import fixed_number_basis, lift_one_body

    b = fixed_number_basis(20, 16)
    p = cluster_inversion(b)
    t = nominal_spin_basis(b)
    assert_allclose(t.conj().T @ t, np.eye(9), atol=1e-14)
    assert_allclose((p @ p).diagonal(), 1, atol=1e-14)
    swap = np.eye(9).reshape(3, 3, 9).transpose(1, 0, 2).reshape(9, 9)
    assert_allclose(p @ t, t @ swap, atol=1e-14)
    positions = point_positions(2.8)
    for axis in range(2):
        mu = lift_one_body(np.diag(-positions[:, axis]), b)
        residual = p @ mu + mu @ p
        assert residual.nnz == 0 or np.max(abs(residual.data)) < 2e-14


def test_response_reduction_preserves_inversion_at_every_size():
    from scipy import sparse

    from nii2_shg.consistent_spiral import shared_response_basis

    rng = np.random.default_rng(84)
    parity = np.r_[np.ones(5), -np.ones(4), np.ones(3), -np.ones(4)]
    p = sparse.diags(parity)
    h = sparse.diags(np.r_[np.zeros(9), np.linspace(1, 4, 7)]).astype(complex)
    operators = []
    for _ in range(2):
        a = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
        a = (a + a.conj().T) / 2
        a[parity[:, None] == parity[None, :]] = 0
        operators.append(sparse.csr_matrix(a))
    low = {"energies": np.zeros(9), "vectors": np.eye(16, dtype=complex)[:, :9]}
    q, _ = shared_response_basis(h, operators, low, 16, 1.2, 0.5, p)
    for count in range(9, 17):
        v = q[:, :count]
        assert_allclose(v.conj().T @ v, np.eye(count), atol=1e-13)
        assert_allclose(p @ v, v @ (v.conj().T @ (p @ v)), atol=1e-13)
