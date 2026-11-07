"""Literal Fan nearest-bond electronic H0 reconstruction; not bulk SHG.

Electron convention: 20 active spin orbitals, nominal 16 electrons (two eg
on each Ni; six p on each I). Retains printed SOC and Delta conventions.
Source: doi:10.1038/s41524-026-02184-w, Eqs. 9-14 and Table 2.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import sparse

from .fan_model import (
    local_kanamori,
    nearest_cluster_one_particle,
    polarization_kernel,
)


def fixed_number_basis(orbitals: int, electrons: int) -> np.ndarray:
    """Ascending occupation integers; bit p denotes spin-orbital p."""
    if not 0 <= electrons <= orbitals <= 62:
        raise ValueError("Require 0 <= electrons <= orbitals <= 62")
    return np.array(
        sorted(
            sum(1 << p for p in occupied)
            for occupied in combinations(range(orbitals), electrons)
        ),
        dtype=np.int64,
    )


def lift_one_body(matrix: np.ndarray, basis: np.ndarray) -> sparse.csr_matrix:
    """Construct sum_pq h[p,q] c†_p c_q with explicit fermion signs."""
    index = {int(state): i for i, state in enumerate(basis)}
    terms = [
        (int(p), int(q), matrix[p, q]) for p, q in zip(*np.nonzero(matrix), strict=True)
    ]
    rows, cols, values = [], [], []
    for col, occupation in enumerate(basis):
        state = int(occupation)
        for p, q, value in terms:
            if not state & (1 << q):
                continue
            intermediate = state ^ (1 << q)
            if intermediate & (1 << p):
                continue
            target = intermediate | (1 << p)
            if target not in index:
                raise ValueError("Basis is not closed under the requested operator")
            parity = (state & ((1 << q) - 1)).bit_count()
            parity += (intermediate & ((1 << p) - 1)).bit_count()
            rows.append(index[target])
            cols.append(col)
            values.append((-1) ** parity * value)
    return sparse.coo_matrix(
        (values, (rows, cols)), shape=(len(basis), len(basis)), dtype=complex
    ).tocsr()


def embed_local(
    matrix: np.ndarray, start: int, width: int, basis: np.ndarray
) -> sparse.csr_matrix:
    """Embed a number-conserving even local operator on contiguous orbitals.

    Local Kanamori terms have even fermion parity, so external Jordan-Wigner
    signs cancel. Local odd operators cannot be embedded by this function.
    """
    if matrix.shape != (1 << width, 1 << width):
        raise ValueError("Local operator shape does not match its orbital width")
    index = {int(state): i for i, state in enumerate(basis)}
    by_column = []
    for local in range(1 << width):
        targets = np.flatnonzero(matrix[:, local])
        if any(int(target).bit_count() != local.bit_count() for target in targets):
            raise ValueError("Only number-conserving local operators are supported")
        by_column.append(targets)
    mask = ((1 << width) - 1) << start
    rows, cols, values = [], [], []
    for col, occupation in enumerate(basis):
        state = int(occupation)
        local = (state & mask) >> start
        for target in by_column[local]:
            full = (state & ~mask) | (int(target) << start)
            rows.append(index[full])
            cols.append(col)
            values.append(matrix[target, local])
    return sparse.coo_matrix(
        (values, (rows, cols)), shape=(len(basis), len(basis)), dtype=complex
    ).tocsr()


def printed_cluster_polarization(parameters: dict, component: str) -> np.ndarray:
    """Assemble Eq. 14 into a Hermitian 20x20 operator; units unresolved.

    This is a literal source operator, not a validated optical dipole or Mou's
    current operator. No susceptibility units or material SHG are inferred.
    """
    matrix = np.zeros((20, 20), complex)
    for ni, iodine, direction in [
        (0, 8, "-x"),
        (0, 14, "+y"),
        (4, 14, "+x"),
        (4, 8, "-y"),
    ]:
        kernel = np.kron(
            polarization_kernel(parameters["nearest_neighbor"], component, direction),
            np.eye(2),
        )
        matrix[ni : ni + 4, iodine : iodine + 6] += kernel
        matrix[iodine : iodine + 6, ni : ni + 4] += kernel.conj().T
    return matrix


def build_cluster(parameters: dict, electrons: int = 16) -> dict:
    """Assemble H1p plus both Ni and both ligand interactions, no cavity field.

    No extra double counting, symmetry-breaking field, broadening or inferred
    optical prefactor is added. SOC/Delta conventions remain unresolved.
    Nominal filling 16 is inferred from the source's ionic occupations.
    """
    basis = fixed_number_basis(20, electrons)
    one_particle, labels = nearest_cluster_one_particle(parameters)
    h = lift_one_body(one_particle, basis)
    local = parameters["local"]
    for start, width, species in [
        (0, 4, "Ni"),
        (4, 4, "Ni"),
        (8, 6, "I"),
        (14, 6, "I"),
    ]:
        interaction, _ = local_kanamori(
            width // 2, local[f"U_{species}"], local[f"J_H_{species}"]
        )
        h += embed_local(interaction, start, width, basis)
    h.eliminate_zeros()
    operators = {
        component: lift_one_body(
            printed_cluster_polarization(parameters, component), basis
        )
        for component in ("x", "y")
    }
    return {
        "h0_ev": h,
        "basis": basis,
        "spin_orbital_labels": labels,
        "printed_polarization": operators,
    }


def export_cluster(root: Path) -> dict:
    """Write reconstructible sparse matrices plus explicit interpretation metadata."""
    import json
    from time import perf_counter

    from .fan_model import load_parameters

    start = perf_counter()
    parameters = load_parameters(root)
    cluster = build_cluster(parameters)
    elapsed = perf_counter() - start
    out = root / "outputs"
    out.mkdir(exist_ok=True)
    sparse.save_npz(out / "fan_cluster_h0.npz", cluster["h0_ev"])
    for component, matrix in cluster["printed_polarization"].items():
        sparse.save_npz(out / f"fan_printed_polarization_{component}.npz", matrix)
    np.savez(
        out / "fan_cluster_basis.npz",
        occupations=cluster["basis"],
        spin_orbital_labels=cluster["spin_orbital_labels"],
    )
    one, _ = nearest_cluster_one_particle(parameters)
    # Ni eg even; I p odd. Exchange the two sites of each species.
    perm = list(range(4, 8)) + list(range(4)) + list(range(14, 20)) + list(range(8, 14))
    inversion = np.zeros((20, 20))
    inversion[perm, range(20)] = [1] * 8 + [-1] * 12
    diagnostics = {}
    for key, matrix in [
        ("h_one_particle", one),
        *[
            (f"printed_p_{c}", printed_cluster_polarization(parameters, c))
            for c in ("x", "y")
        ],
    ]:
        transformed = inversion @ matrix @ inversion.T
        diagnostics[key] = {
            "even_parity_residual_max": float(np.max(abs(transformed - matrix))),
            "odd_parity_residual_max": float(np.max(abs(transformed + matrix))),
        }
    h = cluster["h0_ev"]
    hermiticity = h - h.conj().T
    report = {
        "status": "literal interacting local model reconstructed; not validated bulk optical H0",
        "source": "https://doi.org/10.1038/s41524-026-02184-w",
        "source_locators": "Table 2; Methods Eqs. 9-14",
        "active_spin_orbitals": 20,
        "electrons": 16,
        "filling_basis": "Inferred nominal: Ni eg2 on each of two Ni; I p6 on each of two I",
        "fock_dimension": len(cluster["basis"]),
        "h0_nonzeros": h.nnz,
        "h0_energy_unit": "eV",
        "basis_convention": "Ascending occupation integers, bit p=spin orbital p; orbital-major up/down",
        "h0_sparse_storage_bytes": h.data.nbytes + h.indices.nbytes + h.indptr.nbytes,
        "dense_complex128_bytes": len(cluster["basis"]) ** 2 * 16,
        "assembly_time_s": elapsed,
        "assembly_timing_scope": "Single local CPU build including printed operators; excludes disk I/O and diagonalization",
        "hermiticity_residual_max": float(np.max(abs(hermiticity.data)))
        if hermiticity.nnz
        else 0.0,
        "one_particle_inversion_diagnostics": diagnostics,
        "occupation_encoding_system_qubits": 20,
        "compact_sector_dimension_lower_bound_qubits": 13,
        "qubit_caveat": "13 is ceil(log2(4845)), not a constructed efficient oracle; ancillas and preparation are extra",
        "included": [
            "one-particle hopping/Delta/SOC",
            "local Coulomb terms on both Ni and both I",
            "literal printed x/y polarization operators",
        ],
        "not_included": [
            "magnetic domain/ordered-state selection",
            "bulk/interlayer physics",
            "cavity photons",
            "optical gauge/prefactor validation",
            "broadening or SHG solver",
        ],
        "unresolved": [
            "Printed SOC spectrum and adjacent prose have opposite signs/order; source electron/hole convention needs confirmation.",
            "Literal Eq.14 kernels are inversion-even under the same source-basis inversion that preserves H0. A physical electric dipole must be odd. Clarify operator/basis conventions before SHG.",
            "Printed polarization coefficients have unresolved electromagnetic units/prefactors; not dipoles in e Angstrom.",
            "No source-specific optical dephasing/broadening supplied by Fan; borrowing Mou eta would be a new assumption.",
            "Not Mou H0: different orbital truncations, filling and interaction/optical construction.",
        ],
        "material_shg_ready": False,
    }
    (out / "fan_cluster_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
