"""Independent checks of normalization, phases, symmetry, and full response."""

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from nii2_shg.circuits import (
    circuit_product,
    dipole_block,
    filter_block,
    hamiltonian_walk,
    lcu,
    overlap_circuit,
    sample_hadamard,
)
from nii2_shg.model import (
    OpticalModel,
    beta_from_products,
    classical_beta,
    photon_energy,
)


def test_blocks_and_walk():
    model = OpticalModel()
    dipole = dipole_block(model)
    np.testing.assert_allclose(
        dipole.accepted() * dipole.scale, model.dipole, atol=1e-12
    )
    walk = hamiltonian_walk(model)
    np.testing.assert_allclose(walk.conj().T @ walk, np.eye(8), atol=1e-12)
    np.testing.assert_allclose(
        (walk @ walk)[:4, :4], 2 * model.scaled_hamiltonian**2 - np.eye(4), atol=1e-12
    )
    for target in (0, 1, 2):
        block = filter_block(model, target)
        np.testing.assert_allclose(
            block.accepted()[:3, :3] * block.scale,
            np.diag(np.eye(3)[target]),
            atol=1e-12,
        )


def test_coherent_product():
    result, _, _ = circuit_product(OpticalModel())
    np.testing.assert_allclose(result, 0.12, atol=1e-11)


def test_hadamard_preserves_complex_phase():
    # i*I has overlap i. Probabilities alone would lose this phase.
    block = lcu([1j], [np.eye(4)])
    for quadrature, expected in (("real", 0), ("imag", 1)):
        circuit = overlap_circuit([block], quadrature)
        probabilities = Statevector.from_instruction(circuit).probabilities(
            [circuit.num_qubits - 1]
        )
        np.testing.assert_allclose(
            probabilities[0] - probabilities[1], expected, atol=1e-12
        )


@pytest.mark.parametrize("eta", [0, -1, 0.3, 1])
def test_full_response_against_density_matrix(eta):
    model = OpticalModel(eta)
    products = {(2, 1): 0.12 * eta, (1, 2): 0.12 * eta}
    for energy in (0.7, photon_energy(1030), 1.5, 2.0):
        np.testing.assert_allclose(
            beta_from_products(model, products, energy),
            classical_beta(model, energy),
            atol=1e-11,
        )


def test_inversion_and_domain_sign():
    parity = np.diag([1, -1, 1, 1])
    symmetric = OpticalModel(0)
    np.testing.assert_allclose(parity @ symmetric.dipole @ parity, -symmetric.dipole)
    energy = photon_energy(1030)
    assert abs(classical_beta(symmetric, energy)) < 1e-12
    np.testing.assert_allclose(
        classical_beta(OpticalModel(-1), energy),
        -classical_beta(OpticalModel(1), energy),
    )


def test_sampling_and_invalid_inputs():
    result = sample_hadamard(1 + 0j, 1, 100, np.random.default_rng(1))
    assert result.real == 1
    with pytest.raises(ValueError):
        photon_energy(0)
    with pytest.raises(ValueError):
        classical_beta(OpticalModel(), 1, 0)
    with pytest.raises(ValueError):
        filter_block(OpticalModel(), 4)
    with pytest.raises(ValueError):
        sample_hadamard(0, 1, 0, np.random.default_rng(1))
