"""Validate source extraction and independent tensor projection identities."""

from pathlib import Path

import numpy as np

from nii2_shg.gao_reference import (
    calibrated_patterns,
    read_static_source,
    static_fit_comparison,
)

ROOT = Path(__file__).resolve().parents[1]


def test_preserved_source_layout():
    data = read_static_source(ROOT)
    assert len(data["angle_deg"]) == 73 and len(data["traces"]) == 8
    assert data["angle_deg"][0] == 0 and data["angle_deg"][-1] == 355
    assert data["angle_deg"].count(55.0) == 2
    assert data["traces"][0]["temperature_k"] == 1.63
    assert data["traces"][0]["parallel_uv"][0] == 2.567
    assert data["traces"][0]["crossed_uv"][0] == 80.13


def test_tensor_projection():
    chi = {"xxy": -14.6, "yxx": 8.6, "yyy": 19.7}
    par, cross = calibrated_patterns(np.array([0.0, 90.0, 180.0]), chi)
    np.testing.assert_allclose(par, [0, 19.7**2, 0], atol=1e-10)
    np.testing.assert_allclose(cross, [8.6**2, 0, 8.6**2], atol=1e-10)
    # Both orthogonal projections preserve the norm of the nonlinear vector.
    phi = np.linspace(0, np.pi, 31)
    c, s = np.cos(phi), np.sin(phi)
    par, cross = calibrated_patterns(np.rad2deg(phi), chi)
    np.testing.assert_allclose(
        par + cross,
        (-29.2 * c * s) ** 2 + (8.6 * c * c + 19.7 * s * s) ** 2,
        atol=1e-10,
    )


def test_reconstruction_reports_its_fitted_scale():
    result = static_fit_comparison(ROOT)
    assert result["detector_scale_uv_per_chi_squared"] > 0
    assert np.isfinite(result["rmse_uv"])
    assert "not a quantum prediction" in result["interpretation"]
