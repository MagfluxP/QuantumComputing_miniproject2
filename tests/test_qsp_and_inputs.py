"""Check phase synthesis, a circuit's spectral action, and missing-data behavior."""

from pathlib import Path

import numpy as np
import pytest

from nii2_shg.model import OpticalModel
from nii2_shg.nii2_inputs import MissingNiI2Inputs, input_status, require_nii2_inputs
from nii2_shg.qsp import design_window, qsp_filter, qsp_product, scalar_qsp


@pytest.mark.parametrize("degree", [4, 8, 16, 32])
def test_phase_sequence_against_polynomial(degree):
    design = design_window(degree)
    # Uniform nodes differ from the synthesis Chebyshev nodes.
    x = np.linspace(-1, 1, 233)
    np.testing.assert_allclose(
        scalar_qsp(x, design.phases), design.polynomial(x), atol=1e-6
    )
    assert np.max(abs(design.polynomial(x))) <= 1 + 1e-8


def test_matrix_filter_and_full_overlap():
    model = OpticalModel()
    block, design = qsp_filter(model, 1.5, 32)
    x = (model.energies - 1.5) / max(abs(model.energies - 1.5))
    np.testing.assert_allclose(
        block.accepted() * block.scale, np.diag(design.polynomial(x)), atol=1e-6
    )
    result, _, _ = qsp_product(model, degree=32)
    assert abs(result - 0.12) < 1e-4
    symmetric, _, _ = qsp_product(OpticalModel(0), degree=8)
    assert abs(symmetric) < 1e-10


def test_missing_material_data_never_produces_fake_response():
    root = Path(__file__).resolve().parents[1]
    status = input_status(root)
    assert "hamiltonian" in status["missing"]
    with pytest.raises(MissingNiI2Inputs, match="hamiltonian"):
        require_nii2_inputs(root)


def test_invalid_qsp_design():
    with pytest.raises(ValueError):
        design_window(5)
