# Quantum algorithm for SHG: Gao-state hybrid

Open **[notebooks/01_shg_walkthrough.ipynb](notebooks/01_shg_walkthrough.ipynb)**.
The executed notebook contains one illustrative model and one material model:

1. **Illustrative QSP method:** dipole block encoding, fitted spectral windows,
   coherent products and independent classical second-order response.
2. **Current material calculation:** Gao's S=1 spin Hamiltonian and a weak domain
   selector at 1.63 K determine correlated pair densities. The retained electronic
   energies/dipoles supply the optical response at 1030 nm.
3. **Comparison:** identical-input circuit accuracy, optical convergence,
   experimental disagreement, controlled sensitivities, matched computational
   cost and the conditions for potential quantum advantage.
4. **Summary:** demonstrated results and unresolved material/scaling limits.

Superseded material calculations and their notebook sections/curves are removed.
Source papers, parameter provenance, reusable tested operators and the current
model's uncertainty controls are retained.

## Interpretation

This is a declared hybrid, not an author-validated optical NiI2 calculation.
Gao supplies the **magnetic** H0/state; Fan's electronic parameters remain in
optical denominators. Spin excitation energies are not substituted into optical
resolvents. Fixed primary settings are Table S1 J2=-0.32 meV, T=1.63 K,
h_chi=0.05 meV, q=0.138, normal tilt +55 degrees and one handedness.

The 16-state circuit matches the identical classical calculation, but its optical
reduction is not converged. The 512-state classical reference gives **64.69%**
all-data angular residual and **77.33%** blocked held-out error against Gao's
static data. Absolute detector gain is fitted out; this is not bulk chi in pm/V.

**No computational quantum advantage is demonstrated.** The matched classical
small-instance calculation is faster than CPU statevector simulation. Classical
preprocessing, dense gates, state loading and measurement precision must be
included in any future advantage claim. Section 3 distinguishes this outcome
from the conditional scaling motivation of fault-tolerant spectral filtering.

## Reproduce

Python 3.12 and [uv](https://docs.astral.sh/uv/); the lockfile fixes dependencies:

```sh
uv sync --locked
PYTHONPATH=src uv run python scripts/calculate_hybrid_shg.py
uv run python scripts/benchmark_hybrid.py
uv run python scripts/build_notebook.py
uv run python scripts/execute_notebook.py
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

The calculator includes classical magnetic/electronic preprocessing, response
spaces through 512 states, state sensitivities and eighteen ideal 16-state optical
circuits. The benchmark compares the same preloaded 18 responses in one process.
The notebook recomputes the Gao density, verifies the saved pair mapping and
reruns a quantum/Hadamard/classical optical witness.

`tmp/` is an ignored, disposable numerical cache. No revision-backup folders are
created. Source data under `data/references/raw/` are preserved without edits.

## Files

- [Model construction](docs/hybrid_gao_optical.md)
- [Gao H0 definition](docs/gao_spin_h0.md)
- [Sources and conventions](docs/sources_and_conventions.md)
- `scripts/calculate_hybrid_shg.py`: single material workflow.
- `scripts/benchmark_hybrid.py`: matched CPU and readout-cost diagnostics.
- `scripts/build_notebook.py`, `scripts/material_sections.py`: notebook generation.
- `scripts/execute_notebook.py`: fresh exact-interpreter kernel execution.
- `outputs/hybrid_shg_report.json`, `hybrid_pair_states.npz`, `hybrid_circuits/`:
  current model, states and circuit inputs/readouts.
- `outputs/hybrid_benchmark.json`: measured cost and stated sampling assumptions.

The raw Gao workbook retains its published duplicate 55-degree row. Parallel and
crossed labels remain those in the source. J2's main/supplement sign discrepancy,
axis wording and single-ion convention are documented rather than silently
resolved by fitting the experimental lobes.

## Open the notebook

After `uv sync --locked`, run:

```sh
uv run jupyter lab notebooks/01_shg_walkthrough.ipynb
```

Choose the project's Python 3.12 `.venv` kernel and restart/run all cells. In VS
Code, use **Select Kernel → Python Environments → .venv/bin/python**. The first
cell prints the source path and refreshes this project's imports. The saved
notebook includes outputs; GitHub displays it without running Python.

## Supported inputs and limits

- Toy: four electronic levels (two system qubits), the supplied dipole and spectral
  windows, and even QSP degrees 2–64 with numerical phase fitting. Changing a
  degree or window requires checking its reported approximation/phase-fit error.
- Dense response: K electronic energies, Hermitian K×K input/output dipoles,
  positive broadening, and an optional stationary K×K density. The circuit path
  requires K to be a power of two; the current material circuit uses K=16.
- Magnetic state: the declared seven-site open S=1 cluster, nonnegative temperature,
  chirality selector and either handedness. A truncated thermal calculation must
  pass the reported omitted-weight bound. It is not a bulk thermodynamic solution.
- The optical coupling is a retained point-orbital approximation. No absolute
  susceptibility calibration, pump dynamics, converged material quantum circuit,
  hardware performance or demonstrated speedup is claimed.

## Review and submission

See [requirement checklist](docs/submission_checklist.md) for evidence, remaining
submission actions and the deadline; [GitHub instructions](docs/github.md) explain
publication. The versioned paper scope, library credits, numerical tolerances,
register ordering, compiler settings, gate/query counts and shot assumptions are
explained in the notebook. Public package functions have type hints and docstrings.

Downloaded paper PDFs/archives and extracted manuscripts stay local and are
ignored by Git. Reproduction uses the included source-data workbook, parameter
files and saved computational outputs; provenance URLs/hashes remain included.
