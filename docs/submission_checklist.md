# Mini-project 2 requirement audit

Requirement source: the supplied three-page Erdős Institute bootcamp handout.
Audit date: 7 November 2026. The user confirmed proposal/scope agreement with
Vyom. Final technical verification results are recorded below after execution.

| Requirement | Evidence / status |
|---|---|
| Pre-start proposal and scope agreement | Confirmed by user. |
| Research method beyond class/practice | Notebook paper-scope paragraph: custom LCU/QSP spectral filtering and nonlinear overlaps; material Hamiltonian simulation alone is not the chosen method. |
| Defining method implemented, supporting libraries identified | `src/nii2_shg/circuits.py`, `qsp.py`; versioned sections and explicit library credits in notebook. |
| Clear problem, assumptions, simplifications | Notebook Parts 1–2; `docs/hybrid_gao_optical.md`, `gao_spin_h0.md`. |
| Math, demo, results in notebook; reusable package | `notebooks/01_shg_walkthrough.ipynb`, `src/nii2_shg`, `tests`, `scripts`. |
| Descriptive names, type hints, docstrings | Public package functions typed; shapes/assumptions documented. |
| Register order, ancillas, phase and failure handling | Circuit docstrings/notebook; complex Hadamard tests and all-input coherent-flag test. Fresh work flags are not assumed unconditionally clean. |
| uv, locked dependencies, Python version | `pyproject.toml`, `uv.lock`, `.python-version`. |
| README setup, tests, notebook launch, supported inputs, limits | README Reproduce/Open/Supported inputs sections. |
| Known examples, boundaries, independent references, justified tolerances | Tests; notebook Part 3.1 tolerance explanation. |
| Restart/run-all with outputs and fixed seeds | Fresh-kernel execution; verification result below. |
| Relevant parameter sweeps and matched baselines | QSP degree/window/readout sweeps; same-input circuit/classical checks; optical convergence and experimental held-out/constant baselines. |
| Simulator, shots, gate basis, connectivity and compilation | Part 3.5.1; `outputs/circuit_resources.json`. Ideal CPU Statevector, no executed measurement shots; binomial emulation labeled. |
| Quantum resources separate from preprocessing/simulation | Gate/query/qubit reports versus `outputs/hybrid_benchmark.json`; no physical gate count invented for large dense material blocks. |
| Input preparation, oracles and repetition costs | LCU PREP/SELECT/UNPREP counted; dense construction and state loading explicit; postselection 1/p distinct from Hadamard variance. |
| Citations, version/sections, modifications and libraries | Notebook opening credits and Parts 1–2; source/provenance documents. |
| Claims supported by results | Part 3 states experimental disagreement, unconverged quantum reduction and no demonstrated quantum speedup. |
| GitHub repository and link submitted | **Remaining user action:** publish repository and send link. See `github.md`. |

The deadline is **8 November 2026, 11:59 PM EST**, equivalent to
**10:59 PM Chicago time (CST)** on the same day.

## Verification

Verified on 7 November 2026:

- `uv sync --locked`: passed (118 resolved / 116 installed packages checked).
- `uv run pytest`: **58 passed**.
- `uv run ruff check .`: passed.
- `uv run ruff format --check .`: passed.
- Fresh kernel: **17/17 code cells executed**, saved outputs, zero cell errors.
- GitHub-ready copy: all 17 cells also executed with local paper PDFs/archives,
  manuscript copies and `.venv` excluded. Verification used the installed locked
  interpreter; the notebook loaded its source from the copied tree.
- All public package function annotations resolve successfully.
- Repaired moved `.venv` launchers by reinstalling the same locked dependencies;
  pytest and Jupyter now reference the current project folder.

Compilation reports include raw QSP degree-dependent CX/depth, component counts,
qubits and postselection probability. Large dense material physical synthesis is
explicitly outside the measured scope; no hardware gate/shot count is inferred.
