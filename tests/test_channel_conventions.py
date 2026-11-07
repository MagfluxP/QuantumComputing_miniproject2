"""Independent reference and physical-polarizer checks against channel reversal."""

from pathlib import Path

import numpy as np

from nii2_shg.angular_response import patterns
from nii2_shg.gao_reference import calibrated_patterns, read_static_source

ROOT = Path(__file__).resolve().parents[1]


def test_workbook_labels_include_both_channels():
    source = read_static_source(ROOT)
    assert source["channel_headers"]["parallel"] == "Parallel configuration Signal (uV)"
    assert source["channel_headers"]["crossed"] == "Crossed configuration Signal (uV)"
    assert source["first_temperature_signal_columns"] == {
        "parallel": "B",
        "crossed": "K",
    }


def test_published_tensor_matches_paper_equations_with_same_labels():
    chi = {"xxy": -14.6, "yxx": 8.6, "yyy": 19.7}
    t = np.zeros((2, 2, 2), complex)
    t[0, 0, 1] = t[0, 1, 0] = chi["xxy"]
    t[1, 0, 0] = chi["yxx"]
    t[1, 1, 1] = chi["yyy"]
    phi = np.linspace(0, 360, 83)
    np.testing.assert_allclose(
        patterns(t, phi), np.column_stack(calibrated_patterns(phi, chi)), atol=1e-10
    )


def test_known_polarizer_axes_and_consistent_frame_rotation():
    t = np.zeros((2, 2, 2), complex)
    t[0, 0, 0] = 2
    t[1, 0, 0] = 3
    # At input x, a physical x analyzer is parallel and a y analyzer is crossed.
    np.testing.assert_allclose(patterns(t, [0]), [[4, 9]], atol=1e-12)
    rng = np.random.default_rng(11)
    t = rng.normal(size=(2, 2, 2)) + 1j * rng.normal(size=(2, 2, 2))
    angle = 37.0
    rad = np.deg2rad(angle)
    r = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    rotated = np.einsum("ia,abc,jb,kc->ijk", r, t, r, r)
    phi = np.linspace(0, 360, 59)
    # Rotating tensor and analyzer/input axes together preserves channel names.
    np.testing.assert_allclose(
        patterns(rotated, phi + angle), patterns(t, phi), atol=1e-11
    )
