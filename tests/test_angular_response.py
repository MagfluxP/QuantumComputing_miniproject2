"""Mixed tensor, circuit readout, contractions and grouped holdout checks."""

import numpy as np

from nii2_shg.angular_response import (
    CUBIC_TO_PLANE,
    angular_comparison,
    classical_tensor,
    fit_scale_offset,
    patterns,
)
from nii2_shg.material_response import classical_response
from nii2_shg.response_circuit import quantum_response


def test_mixed_tensor_and_quantum_output_operator():
    rng = np.random.default_rng(32)
    operators = []
    for _ in range(2):
        m = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        operators.append((m + m.conj().T) / 5)
    energies = np.array([0.0, 1.3, 2.7, 4.1])
    tensor = classical_tensor(energies, operators, 1.2, 0.5)
    angle = 37.0
    e = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
    cross = np.array([-e[1], e[0]])
    mu = e[0] * operators[0] + e[1] * operators[1]
    output = cross[0] * operators[0] + cross[1] * operators[1]
    expected = classical_response(energies, mu, 1.2, 0.5, output_mu=output)
    q = quantum_response(energies, mu, 1.2, 0.5, output_mu=output)
    np.testing.assert_allclose(q["beta"], expected, atol=1e-11)
    np.testing.assert_allclose(q["hadamard_beta"], expected, atol=1e-10)
    np.testing.assert_allclose(
        patterns(tensor, [angle])[0, 1], abs(expected) ** 2, atol=1e-12
    )
    np.testing.assert_allclose(tensor[:, 0, 1], tensor[:, 1, 0])
    np.testing.assert_allclose(CUBIC_TO_PLANE @ CUBIC_TO_PLANE.T, np.eye(2), atol=1e-12)


def test_common_scale_and_periodic_orientation():
    rng = np.random.default_rng(5)
    t = rng.normal(size=(2, 2, 2)) + 1j * rng.normal(size=(2, 2, 2))
    t = (t + t.swapaxes(1, 2)) / 2
    angles = np.arange(0, 360, 5.0)
    y = 17 * patterns(t, angles, 23.4)
    fit = fit_scale_offset(t, angles, y)
    np.testing.assert_allclose(
        fit["common_scale"] * patterns(t, angles, fit["offset_deg"]), y, atol=1e-5
    )
    np.testing.assert_allclose(
        patterns(t, angles), patterns(t, angles + 180), atol=1e-12
    )


def test_holdout_keeps_channels_duplicates_and_180_counterparts_together():
    t = np.zeros((2, 2, 2), complex)
    t[0, 0, 0] = 1
    t[1, 0, 1] = t[1, 1, 0] = 0.8
    t[1, 1, 1] = 0.2j
    angles = np.r_[np.arange(0, 360, 5.0), 55.0]
    y = 10 * patterns(t, angles, 12.0)
    result = angular_comparison(t, angles, y)
    folds = np.array(result["fold_index"])
    assert folds[np.flatnonzero(angles == 55)[0]] == folds[-1]
    for a in range(0, 180, 5):
        assert (
            folds[np.flatnonzero(angles == a)[0]]
            == folds[np.flatnonzero(angles == a + 180)[0]]
        )
    assert result["out_of_fold_metrics"]["joint"]["relative_l2"] < 1e-6
    # Altering one whole held-out block must not change its fitted predictions.
    changed = y.copy()
    changed[folds == 2] += 100
    other = angular_comparison(t, angles, changed)
    np.testing.assert_allclose(
        np.array(result["out_of_fold_prediction_uv"])[folds == 2],
        np.array(other["out_of_fold_prediction_uv"])[folds == 2],
        atol=1e-10,
    )


def test_c2_diagnostic_does_not_change_tensor():
    from nii2_shg.angular_response import c2_forbidden_fraction

    t = np.zeros((2, 2, 2), complex)
    t[1, 0, 0] = 1
    t[0, 0, 1] = t[0, 1, 0] = 2
    t[1, 1, 1] = 3
    original = t.copy()
    assert c2_forbidden_fraction(t) == 0
    np.testing.assert_array_equal(t, original)
    t = np.zeros_like(t)
    t[0, 0, 0] = 1
    assert c2_forbidden_fraction(t) == 1
