"""Independent second-quantization and atomic-limit checks for the reconstruction."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from nii2_shg.fan_cluster import (
    build_cluster,
    embed_local,
    fixed_number_basis,
    lift_one_body,
)
from nii2_shg.fan_model import load_parameters

ROOT = Path(__file__).resolve().parents[1]


def test_one_body_particle_and_full_sectors():
    rng = np.random.default_rng(6)
    h = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    h += h.conj().T
    np.testing.assert_allclose(lift_one_body(h, fixed_number_basis(4, 1)).toarray(), h)
    np.testing.assert_allclose(
        lift_one_body(h, fixed_number_basis(4, 4)).toarray(), [[np.trace(h)]]
    )
    # A hop 0 -> 2 passes occupied orbital 1 and must acquire a minus sign.
    h = np.zeros((3, 3))
    h[2, 0] = 1
    b = fixed_number_basis(3, 2).tolist()
    assert lift_one_body(h, np.array(b))[b.index(6), b.index(3)] == -1


def test_local_embedding_and_rejection():
    # n0*n1 on local orbitals [1,2], independent of spectator orbital 0.
    matrix = np.diag([0, 0, 0, 2])
    b = fixed_number_basis(4, 2)
    result = embed_local(matrix, 1, 2, b).toarray()
    np.testing.assert_array_equal(
        np.diag(result), [2 if int(s) & 6 == 6 else 0 for s in b]
    )
    with pytest.raises(ValueError, match="number-conserving"):
        embed_local(np.array([[0, 1], [1, 0]]), 0, 1, b)


def test_atomic_hunds_manifold():
    p = deepcopy(load_parameters(ROOT))
    p["nearest_neighbor"] = dict.fromkeys(p["nearest_neighbor"], 0.0)
    cluster = build_cluster(p)
    assert cluster["h0_ev"].shape == (4845, 4845)
    # Each Ni has E(N=2,S=1)=1; each full I has E(N=6)=15-12=3.
    # Independent site sums give nine product-triplet states at total E=8 eV.
    # Diagonalize the complete 36-state nominal ionic sector. A scalar Krylov
    # solver need not recover every vector in an exactly degenerate eigenspace.
    indices = [
        i
        for i, s in enumerate(cluster["basis"])
        if (int(s) & 15).bit_count() == 2
        and ((int(s) >> 4) & 15).bit_count() == 2
        and ((int(s) >> 8) & 63) == 63
        and ((int(s) >> 14) & 63) == 63
    ]
    assert len(indices) == 36
    spectrum = np.linalg.eigvalsh(cluster["h0_ev"][indices, :][:, indices].toarray())
    np.testing.assert_allclose(spectrum[:9], 8, atol=1e-8)
    assert spectrum[9] > 8.5


def test_reconstructed_operator_symmetries():
    c = build_cluster(load_parameters(ROOT))
    assert all(int(s).bit_count() == 16 for s in c["basis"])
    for matrix in [c["h0_ev"], *c["printed_polarization"].values()]:
        residual = matrix - matrix.conj().T
        assert not residual.nnz or np.max(abs(residual.data)) < 1e-12
    # Source geometry inversion swaps Ni sites with + and ligand sites with -.
    # Apply that permutation, including fermion reordering signs, to the fixed-N basis.
    perm = list(range(4, 8)) + list(range(4)) + list(range(14, 20)) + list(range(8, 14))
    index = {int(s): i for i, s in enumerate(c["basis"])}
    rows, signs = [], []
    for state in c["basis"]:
        occupied = [p for p in range(20) if int(state) & (1 << p)]
        targets = [perm[p] for p in occupied]
        swaps = sum(a > b for i, a in enumerate(targets) for b in targets[i + 1 :])
        ligand_count = sum(p >= 8 for p in occupied)
        rows.append(index[sum(1 << p for p in targets)])
        signs.append((-1) ** (swaps + ligand_count))
    from scipy.sparse import coo_matrix

    inversion = coo_matrix(
        (signs, (rows, range(len(rows)))), shape=c["h0_ev"].shape
    ).tocsr()
    for key, matrix in [("H", c["h0_ev"]), *c["printed_polarization"].items()]:
        # Literal Eq.14 is also even under this source-basis inversion, although
        # a physical electric dipole should be odd. Preserve and flag this
        # unresolved convention instead of editing the published kernels.
        residual = inversion @ matrix @ inversion.conj().T - matrix
        assert not residual.nnz or np.max(abs(residual.data)) < 1e-12
