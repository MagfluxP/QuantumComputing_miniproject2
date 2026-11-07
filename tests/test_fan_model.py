"""Independent physical checks of the transcribed electronic primitives."""

from pathlib import Path

import numpy as np
import pytest

from nii2_shg.fan_model import (
    hopping_kernel,
    ligand_soc,
    load_parameters,
    local_kanamori,
    nearest_cluster_one_particle,
    polarization_kernel,
)

ROOT = Path(__file__).resolve().parents[1]


def test_ligand_soc_multiplet():
    soc = ligand_soc(0.5)
    np.testing.assert_allclose(soc, soc.conj().T, atol=1e-12)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(soc), [-0.5] * 2 + [0.25] * 4, atol=1e-12
    )


def test_ni_hunds_triplet():
    h, n = local_kanamori(2, 4, 1)
    indices = np.flatnonzero(np.isclose(np.diag(n).real, 2))
    np.testing.assert_allclose(
        np.linalg.eigvalsh(h[np.ix_(indices, indices)]), [1, 1, 1, 3, 3, 5], atol=1e-12
    )
    np.testing.assert_allclose(h @ n, n @ h, atol=1e-12)


def test_cluster_and_signs():
    parameters = load_parameters(ROOT)
    h, labels = nearest_cluster_one_particle(parameters)
    assert h.shape == (20, 20) and len(set(labels)) == 20
    np.testing.assert_allclose(h, h.conj().T, atol=1e-12)
    # The centrosymmetric one-particle cluster cannot represent a selected helix.
    inversion = np.zeros_like(h)
    for a, b, dimension, sign in [(0, 4, 4, 1), (8, 14, 6, -1)]:
        inversion[a : a + dimension, b : b + dimension] = sign * np.eye(dimension)
        inversion[b : b + dimension, a : a + dimension] = sign * np.eye(dimension)
    np.testing.assert_allclose(inversion @ h @ inversion, h, atol=1e-12)
    hop = parameters["nearest_neighbor"]
    np.testing.assert_allclose(hopping_kernel(hop, "-x"), -hopping_kernel(hop, "+x"))
    np.testing.assert_allclose(
        h[:4, 8:14], np.kron(hopping_kernel(hop, "-x"), np.eye(2))
    )


def test_polarization_convention_and_invalid_input():
    hop = load_parameters(ROOT)["nearest_neighbor"]
    np.testing.assert_allclose(
        polarization_kernel(hop, "x", "+x"),
        1j * np.array([[0, 0.77, 0], [0, -0.44, 0]]),
    )
    assert not np.any(polarization_kernel(hop, "x", "+y"))
    with pytest.raises(ValueError):
        hopping_kernel(hop, "bad")
    with pytest.raises(ValueError):
        polarization_kernel(hop, "z", "+x")
    with pytest.raises(ValueError):
        local_kanamori(5, 4, 1)
