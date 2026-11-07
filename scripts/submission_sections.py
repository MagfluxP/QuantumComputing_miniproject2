"""Assignment-specific scope, credits, tolerances and reproducible resources."""

from textwrap import dedent

import nbformat as nbf


def add_submission_sections(cells: list[nbf.NotebookNode]) -> list[nbf.NotebookNode]:
    intro = nbf.v4.new_markdown_cell(
        dedent(r"""
    ### Paper scope, versions and library credits

    **Chosen method:** Kharazi, Stetina, Ko, Low and Whaley, *An efficient quantum
    algorithm for generation of ab initio n-th order susceptibilities for non-linear
    spectroscopies*, **arXiv:2404.01454v1 (1 April 2024)**. Implementation references
    are to that version: II.3 (LCU), II.4 (QSP), III.1.2 (spectral indicators),
    III.1.3 (non-unitary Hadamard test), and III.2 (higher-order products).
    [Versioned source](https://arxiv.org/html/2404.01454v1).

    **What is implemented here:** custom Pauli-LCU assembly, bounded polynomial
    construction, numerical phase fitting, repeated-signal QSP circuits, coherent
    dipole/filter products and complex readout on a small optical example. Window
    locations and the ground state are supplied; adaptive spectral discovery and
    fault-tolerant chemistry oracles are omitted. The Gao-state hybrid is a project
    extension using dense resolvent blocks, not full-space execution of the paper.
    Its magnetic and electronic parameter transcriptions are credited in Part 2.
    The project is a spectral-filter/nonlinear-response construction, not merely
    the Hamiltonian-simulation exercise covered in class.

    **Library support:** Qiskit supplies QuantumCircuit, gates, StatePreparation,
    Pauli decomposition, Operator/Statevector simulation and transpilation. NumPy
    supplies matrix/polynomial arithmetic; SciPy supplies least-squares phase
    optimization, sparse eigensolvers/LU factorizations, matrix exponentials and
    special functions. Matplotlib/IPython handle plots and presentation. The
    project assembles the chosen method; it does not call a complete built-in
    QSP/nonlinear-spectroscopy algorithm. No third-party project implementation
    is vendored; mathematical definitions and transcribed matrices follow the
    cited papers.
    """).strip()
    )
    cells.insert(1, intro)
    index = next(
        i
        for i, c in enumerate(cells)
        if c.cell_type == "markdown" and "## 3.1 Numerical correctness" in c.source
    )
    cells.insert(
        index + 1,
        nbf.v4.new_markdown_cell(
            dedent(r"""
    **Tests and numerical tolerances.** Pytest checks known matrix products,
    complex Hadamard phase, accepted block normalization, symmetry controls,
    frame/site ordering, invalid inputs and independent density-matrix response.
    A dedicated all-input operator test checks that fresh ancilla flags prevent a
    previously rejected history from returning to the accepted branch. Work
    ancillas are zero only in the accepted branch; unconditional cleanup of a
    non-unitary block is neither assumed nor claimed.

    Float64 exact small-matrix identities use absolute tolerances around 1e−12
    (1e−11–2e−10 after chained response/readout normalization). Tiny spin-frame
    identities use 1e−14; nearly saturated SVD square-root dilations allow 3e−8
    unitarity error because square roots amplify roundoff near zero. QSP
    phase-to-polynomial checks allow 1e−6 and report filter approximation error
    separately. Recomputed magnetic pair densities allow 1e−7 against saved data
    to accommodate degenerate-subspace numerics. These are numerical tolerances,
    not experimental uncertainties or proofs of model accuracy. Absolute checks
    are used for symmetry-zero signals instead of relative errors on noise.
    """).strip()
        ),
    )
    index = next(
        i
        for i, c in enumerate(cells)
        if c.cell_type == "markdown" and "## 3.6 What quantum advantage" in c.source
    )
    before, after = (
        cells[index].source.split("    ## 3.6 What quantum advantage means here", 1)
        if "    ## 3.6 What quantum advantage means here" in cells[index].source
        else cells[index].source.split("## 3.6 What quantum advantage means here", 1)
    )
    cells[index].source = before.rstrip()
    resource_md = nbf.v4.new_markdown_cell(
        dedent(r"""
    ### 3.5.1 Gate counts, compilation, oracle work and success probabilities

    Compile the toy repeated-signal circuit and its small supporting components
    to **u/cx**, **all-to-all connectivity**, optimization level **1**, seed **7**.
    Report actual raw repeated-signal CX/depth versus polynomial degree. The
    accepted filter is also synthesized from its small dense matrix: those
    separate counts include LCU PREP/SELECT/UNPREP but can hide query growth.
    They must not be used as scalable oracle costs. Component-sum counts below
    describe the base product, not a fully compiled controlled-Hadamard circuit.
    Ground preparation is |00> only for this supplied toy example.

    Simulator: `qiskit.quantum_info.Statevector`, ideal CPU, **zero executed
    measurement shots**. Material resources are reported at logical block level
    because physical synthesis of the large dense blocks is not attempted.
    Classical oracle construction/state loading are counted separately from
    compiled gate counts and simulator wall time.
    """).strip()
    )
    resource_code = nbf.v4.new_code_cell(
        dedent("""
    from nii2_shg.resources import collect_resources
    from nii2_shg.benchmarks import run_benchmark
    resource_report=collect_resources()
    (OUT/'circuit_resources.json').write_text(json.dumps(resource_report,indent=2))
    lines=['| Degree | Raw signal calls | Raw QSP CX / depth | Base product component-sum CX | Work qubits | Postselection success |','|---:|---:|---:|---:|---:|---:|']
    for row in resource_report['rows']:
        lines.append(f"| {row['degree']} | {row['raw_sequence_signal_calls']} | {row['raw_qsp']['cx']} / {row['raw_qsp']['depth']} | {row['base_product_component_sum_cx']} | {row['product_work_qubits']} | {row['accepted_work_zero_probability']:.6g} |")
    display(Markdown('\\n'.join(lines)))
    row=resource_report['rows'][-1]
    print('Basis / connectivity / optimization / seed:',resource_report['basis_gates'],resource_report['connectivity'],resource_report['optimization_level'],resource_report['seed_transpiler'])
    print('Toy system / product total / Hadamard total qubits:',row['product_system_qubits'],row['product_total_qubits'],row['hadamard_total_qubits'])
    print('Two-filter unoptimized controlled-signal count:',row['two_filters_unoptimized_controlled_signal_calls'])
    print('Expected attempts IF generating a postselected state:',row['expected_attempts_if_postselecting_state'])
    print('Compilation CPU seconds (raw QSP / dense filters / dipole):',row['raw_qsp']['compile_s'],row['dense_realpart_filter_1']['compile_s']+row['dense_realpart_filter_2']['compile_s'],resource_report['dipole']['compile_s'])
    print('Material logical resources:',resource_report['material_logical_resources'])
    toy_benchmark=run_benchmark(degrees=(32,),repeats=3)
    (OUT/'toy_benchmark.json').write_text(json.dumps(toy_benchmark,indent=2))
    sample_rows=toy_benchmark['sampling']
    fig,ax=plt.subplots(figsize=(6.4,3.2))
    ax.loglog([r['total_shots'] for r in sample_rows],[r['rms_statistical_error'] for r in sample_rows],'o-')
    ax.set(xlabel='Total emulated Hadamard samples (both quadratures)',ylabel='RMS complex-product error (model units)',title='Toy readout: 400 seeded binomial repetitions per budget')
    fig.tight_layout();fig.savefig(OUT/'sampling_resources.png',dpi=160);plt.show()
    """).strip()
    )
    resource_note = nbf.v4.new_markdown_cell(
        dedent(r"""
    The sampling sweep is **classical binomial emulation** from circuit-derived
    probabilities (seed 7), not a hardware or shot-backend run. N samples per
    real/imaginary quadrature cost 2N executions. The material witness's analytic
    variance estimate above concerns the current dense readout, not an optimized
    fault-tolerant resource bound.

    A postselected state-generation procedure would need mean 1/p attempts with
    success p. Hadamard overlap estimation retains failure branches coherently;
    its small-overlap penalty is already in the shot variance and is **not**
    multiplied again by 1/p. Fresh flags preserve phase and prevent failed
    histories from being recycled into success. Compiling a dense SELECT matrix
    is supporting small-instance synthesis, not a free or scalable oracle.
    """).strip()
    )
    cells[index + 1 : index + 1] = [
        resource_md,
        resource_code,
        resource_note,
        nbf.v4.new_markdown_cell("## 3.6 What quantum advantage means here" + after),
    ]
    return cells
