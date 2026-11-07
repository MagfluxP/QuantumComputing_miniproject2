"""Single-domain tensor contractions and held-out angular-shape scoring.

No tensor coefficient is fitted to experiment. Only one nonnegative common
intensity scale and a shared azimuthal offset are nuisance fit parameters.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

# Rows are assumed lab x,y in orthonormal local cubic x,y,z coordinates.
# c || (111), x along the nearest Z bond (-1,1,0), y=c cross x.
CUBIC_TO_PLANE = np.array([[-1, 1, 0], [-1, -1, 2]], float) / np.sqrt([[2], [6]])


def project_dipoles(
    h: Any, dipoles: Sequence[Any], basis: np.ndarray, size: int
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Project all polarizations with a single common eigenbasis."""
    q = basis[:, :size]
    hp = q.conj().T @ (h @ q)
    energies, u = np.linalg.eigh((hp + hp.conj().T) / 2)
    result = []
    for op in dipoles:
        m = u.conj().T @ (q.conj().T @ (op @ q)) @ u
        m = (m + m.conj().T) / 2
        result.append(m - m[0, 0] * np.eye(size))
    return energies - energies[0], result


def classical_tensor(
    energies: np.ndarray,
    dipoles: Sequence[np.ndarray],
    photon_ev: float,
    gamma_ev: float,
    initial_rho: np.ndarray | None = None,
    second_gamma_factor: float = 1.0,
) -> np.ndarray:
    """Six independent in-plane tensor components; input indices symmetrized.

    For input (x+y)/sqrt(2), beta_i,aa=(beta_i,xx+beta_i,yy)/2+beta_i,xy.
    Both output components reuse the same independently computed rho2.
    """
    e = np.asarray(energies)
    gap = e[:, None] - e[None, :]
    rho0 = np.zeros((len(e), len(e)), complex)
    rho0[0, 0] = 1
    if initial_rho is not None:
        rho0 = np.asarray(initial_rho, dtype=complex)
    tensor = np.zeros((2, 2, 2), complex)
    values = []
    for mu in [dipoles[0], dipoles[1], (dipoles[0] + dipoles[1]) / np.sqrt(2)]:
        rho1 = -(mu @ rho0 - rho0 @ mu) / (photon_ev - gap + 1j * gamma_ev)
        rho2 = -(mu @ rho1 - rho1 @ mu) / (
            2 * photon_ev - gap + 1j * gamma_ev * second_gamma_factor
        )
        values.append(np.array([np.einsum("ij,ji->", out, rho2) for out in dipoles]))
    tensor[:, 0, 0] = values[0]
    tensor[:, 1, 1] = values[1]
    tensor[:, 0, 1] = tensor[:, 1, 0] = values[2] - (values[0] + values[1]) / 2
    return tensor


def patterns(
    tensor: np.ndarray,
    angles_deg: np.ndarray | Sequence[float],
    offset_deg: float = 0.0,
) -> np.ndarray:
    """Parallel/crossed |beta|^2, same domain/Hamiltonian at every angle."""
    phi = np.deg2rad(np.asarray(angles_deg) - offset_deg)
    e = np.column_stack([np.cos(phi), np.sin(phi)])
    crossed = np.column_stack([-np.sin(phi), np.cos(phi)])
    p = np.einsum("ijk,nj,nk->ni", tensor, e, e)
    return np.column_stack(
        [abs(np.sum(p * e, axis=1)) ** 2, abs(np.sum(p * crossed, axis=1)) ** 2]
    )


def fit_scale_offset(
    tensor: np.ndarray,
    angles: np.ndarray,
    observed: np.ndarray,
    weights: np.ndarray | None = None,
    fit_offset: bool = True,
) -> dict[str, float]:
    """Analytic nonnegative common scale; periodic global azimuth search.

    Fit only on passed samples. Weights act on angular rows, equally on both
    channels. Offset is periodic modulo 180 degrees for intensity-only SHG.
    """
    angles = np.asarray(angles)
    y = np.asarray(observed)
    w = np.ones(len(angles)) if weights is None else np.asarray(weights)

    def evaluate(offset):
        p = patterns(tensor, angles, offset)
        denominator = np.sum(w[:, None] * p * p)
        scale = (
            max(0.0, float(np.sum(w[:, None] * p * y) / denominator))
            if denominator > 1e-30
            else 0.0
        )
        residual = scale * p - y
        return float(np.sum(w[:, None] * residual**2)), scale

    if fit_offset:
        grid = np.arange(0.0, 180.0, 0.5)
        scores = np.array([evaluate(theta)[0] for theta in grid])
        minima = np.flatnonzero(
            (scores <= np.roll(scores, 1)) & (scores <= np.roll(scores, -1))
        )
        results = [
            minimize_scalar(
                lambda x: evaluate(x % 180)[0],
                bounds=(grid[i] - 0.5, grid[i] + 0.5),
                method="bounded",
                options={"xatol": 1e-8},
            )
            for i in minima
        ]
        best = min(results, key=lambda r: r.fun)
        offset = float(best.x % 180)
    else:
        offset = 0.0
    loss, scale = evaluate(offset)
    return {"offset_deg": offset, "common_scale": scale, "training_sse": loss}


def error_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    y = np.asarray(observed)
    r = np.asarray(predicted) - y

    def metric(a, b):
        residual = b - a
        return {
            "rmse_uv": float(np.sqrt(np.mean(residual**2))),
            "relative_l2": float(
                np.linalg.norm(residual) / max(np.linalg.norm(a), 1e-30)
            ),
        }

    return {
        "joint": metric(y, predicted),
        "parallel": metric(y[:, 0], predicted[:, 0]),
        "crossed": metric(y[:, 1], predicted[:, 1]),
        "residual_uv": r.tolist(),
    }


def angular_comparison(
    tensor: np.ndarray,
    angles: np.ndarray,
    observed: np.ndarray,
    duplicate_equal_weight: bool = False,
) -> dict[str, Any]:
    """Six held-out contiguous blocks, folded modulo 180° to prevent leakage.

    Every sample is predicted by a fit excluding its whole 30-degree angular
    block, its 180-degree counterpart, all duplicates and both channels.
    Report OOF scores separately from a descriptive all-data fit.
    """
    angles = np.asarray(angles)
    y = np.asarray(observed)
    folded = np.mod(angles, 180)
    folds = np.floor(folded / 30).astype(int)
    weights = np.ones(len(angles))
    if duplicate_equal_weight:
        _, inverse, counts = np.unique(
            np.mod(angles, 360), return_inverse=True, return_counts=True
        )
        weights = 1 / counts[inverse]
    predicted = np.zeros_like(y)
    results = []
    for fold in range(6):
        test = folds == fold
        train = ~test
        fit = fit_scale_offset(tensor, angles[train], y[train], weights[train])
        predicted[test] = fit["common_scale"] * patterns(
            tensor, angles[test], fit["offset_deg"]
        )
        results.append(
            {
                "fold": fold,
                "held_out_folded_interval_deg": [30 * fold, 30 * (fold + 1)],
                "train_row_count": int(train.sum()),
                "test_row_count": int(test.sum()),
                "fit": fit,
                "test_metrics": error_metrics(y[test], predicted[test]),
            }
        )
    full = fit_scale_offset(tensor, angles, y, weights)
    full_prediction = full["common_scale"] * patterns(
        tensor, angles, full["offset_deg"]
    )
    # Peak diagnostics use the physical full-fit pattern, rather than discontinuous OOF curves.
    grid = np.linspace(0, 180, 1801, endpoint=False)
    peaks = patterns(tensor, grid, full["offset_deg"])
    return {
        "interpretation": "Single-domain local-bond shape comparison; absolute amplitude fitted out; not bulk validation",
        "fit_parameters": [
            "one common nonnegative detector scale",
            "one common azimuth offset modulo 180 degrees",
        ],
        "cross_validation": "6 contiguous 30-degree blocks modulo 180; duplicate angles and both channels grouped",
        "absolute_amplitude_tested": False,
        "domain_mixture_fitted": False,
        "physical_parameters_fitted": False,
        "duplicate_fit_weighting": "equal total weight per exact angle"
        if duplicate_equal_weight
        else "each published row has equal weight",
        "out_of_fold_metrics": error_metrics(y, predicted),
        "out_of_fold_prediction_uv": predicted.tolist(),
        "folds": results,
        "fold_index": folds.tolist(),
        "all_data_fit": full,
        "all_data_metrics": error_metrics(y, full_prediction),
        "all_data_prediction_uv": full_prediction.tolist(),
        "parallel_peak_angle_mod180_deg": float(grid[np.argmax(peaks[:, 0])]),
        "crossed_peak_angle_mod180_deg": float(grid[np.argmax(peaks[:, 1])]),
        "parallel_to_crossed_peak_ratio": float(
            peaks[:, 0].max() / max(peaks[:, 1].max(), 1e-30)
        ),
    }


def c2_forbidden_fraction(tensor: np.ndarray, offset_deg: float = 0.0) -> float:
    """Descriptive C2-about-lab-y diagnostic; never projects or fits the tensor.

    Under C2_y, x changes sign and y does not. Rank-three electric tensor
    components with an odd number of x indices must vanish if this is a symmetry.
    """
    phi = np.deg2rad(offset_deg)
    r = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    rotated = np.einsum("ia,abc,jb,kc->ijk", r, tensor, r, r)
    forbidden = np.zeros((2, 2, 2), bool)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                forbidden[i, j, k] = ((i == 0) + (j == 0) + (k == 0)) % 2 == 1
    return float(
        np.linalg.norm(rotated[forbidden]) / max(np.linalg.norm(rotated), 1e-30)
    )
