"""Hybrid Gao spin density / fixed Fan-model optical operator and circuits.

Classical spin diagonalization and optical preprocessing are explicit. Only
same-input 16-state circuit agreement is arithmetic validation. No fit of physical
parameters or transfer of Gao spin energies into optical resolvents occurs.
"""

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nii2_shg.angular_response import angular_comparison, patterns
from nii2_shg.consistent_spiral import (
    build_unpinned,
    effective_optical_tensor,
    low_manifold,
    nominal_spin_basis,
    project_unpinned,
    shared_response_basis,
)
from nii2_shg.gao_reference import read_static_source
from nii2_shg.gao_spin_model import GaoSpinConfig, build_gao_h0
from nii2_shg.hybrid_response import (
    HybridStateConfig,
    hybrid_tensor,
    oriented_pair_states,
    selected_state,
)

PHOTON = 1239.841984 / 1030


def serial(v):
    if isinstance(v, np.ndarray):
        return (
            {"real": v.real.tolist(), "imag": v.imag.tolist()}
            if np.iscomplexobj(v)
            else v.tolist()
        )
    if isinstance(v, complex):
        return {"real": v.real, "imag": v.imag}
    if isinstance(v, np.generic):
        return serial(v.item())
    if isinstance(v, dict):
        return {k: serial(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [serial(x) for x in v]
    return v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--reuse-optical-cache", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out or root / "outputs"
    out.mkdir(exist_ok=True, parents=True)
    cache = args.cache or root / "tmp/hybrid"
    cache.mkdir(exist_ok=True, parents=True)
    start = perf_counter()
    config = HybridStateConfig()
    spin = build_gao_h0(root)
    state = selected_state(spin, config)
    pairs = oriented_pair_states(spin, state)
    source = read_static_source(root)
    angles = np.array(source["angle_deg"])
    observed = np.column_stack(
        [source["traces"][0]["parallel_uv"], source["traces"][0]["crossed_uv"]]
    )
    optical = build_unpinned(root)
    h = optical["h0_ev"]
    mus = optical["dipoles"]
    ionic = nominal_spin_basis(optical["basis"])
    saved = cache / "optical_basis.npz"
    import hashlib

    signature = hashlib.sha256(
        (root / "data/nii2/fan2026_parameters.json").read_bytes()
        + (root / "src/nii2_shg/consistent_spiral.py").read_bytes()
        + (root / "src/nii2_shg/fan_model.py").read_bytes()
        + (root / "src/nii2_shg/fan_cluster.py").read_bytes()
        + b"hybrid-1030-gamma0.5-max512-v1"
    ).hexdigest()
    if args.reuse_optical_cache and saved.exists():
        data = np.load(saved)
        if str(data["signature"]) != signature:
            raise ValueError("Optical cache is stale")
        q = data["q"]
        basis_s = float(data["preprocessing_s"])
    else:
        low = low_manifold(h, optical["basis"])
        q, basis_s = shared_response_basis(
            h, mus, low, 512, PHOTON, 0.5, optical["inversion"]
        )
        np.savez_compressed(saved, q=q, preprocessing_s=basis_s, signature=signature)
    if np.max(abs(q.conj().T @ q - np.eye(512))) > 1e-10:
        raise ValueError("Invalid optical basis")
    rows = []
    spaces = {}
    for size in [16, 32, 64, 128, 256, 384, 512]:
        e, m, r, _ = project_unpinned(h, mus, q, size, ionic)
        op = effective_optical_tensor(e, m, PHOTON, 0.5, 2.0)
        tensor = hybrid_tensor(op, r, pairs)
        score = angular_comparison(tensor, angles, observed)
        rows.append({"states": size, "tensor": tensor, "comparison": score})
        if size in [16, 512]:
            spaces[size] = (e, m, r, op)
        print(
            size,
            "hybrid full/heldout",
            score["all_data_metrics"]["joint"]["relative_l2"],
            score["out_of_fold_metrics"]["joint"]["relative_l2"],
            flush=True,
        )
    reference = rows[-1]["tensor"]
    for row in rows:
        row["tensor_relative_difference_to_512"] = float(
            np.linalg.norm(row["tensor"] - reference) / np.linalg.norm(reference)
        )
    e, m, r, op = spaces[512]
    sensitivities = []
    specs = [
        ("No domain selector", replace(config, selector_mev=0), GaoSpinConfig()),
        ("Selector 0.005 meV", replace(config, selector_mev=0.005), GaoSpinConfig()),
        ("Selector 0.25 meV", replace(config, selector_mev=0.25), GaoSpinConfig()),
        ("Opposite handedness", replace(config, handedness=-1), GaoSpinConfig()),
        ("Zero temperature", replace(config, temperature_k=0), GaoSpinConfig()),
        ("Opposite tilt branch", replace(config, normal_tilt_deg=-55), GaoSpinConfig()),
        ("Main-text J2 sign", config, GaoSpinConfig("main_text")),
    ]
    for label, cfg, scfg in specs:
        c = spin if scfg == GaoSpinConfig() else build_gao_h0(root, scfg)
        st = selected_state(c, cfg)
        pr = oriented_pair_states(c, st)
        tensor = hybrid_tensor(op, r, pr)
        relative = float(np.linalg.norm(tensor) / np.linalg.norm(reference))
        # Zero-source residual is a symmetry control, never normalize its noise.
        score = (
            angular_comparison(tensor, angles, observed) if relative > 1e-7 else None
        )
        sensitivities.append(
            {
                "name": label,
                "spin_config": asdict(scfg),
                "state_config": asdict(cfg),
                "tensor": tensor,
                "amplitude_norm_relative_to_primary": relative,
                "thermal_tail_bound": st["thermal_tail_bound"],
                "comparison": score,
            }
        )
        print(
            label,
            "amplitude ratio",
            relative,
            "full error",
            None
            if score is None
            else score["all_data_metrics"]["joint"]["relative_l2"],
            flush=True,
        )
    # Eighteen quantum optical evaluations after loading the Gao-derived densities.
    circuitdir = out / "hybrid_circuits"
    circuitdir.mkdir(exist_ok=True)
    qe, qm, qr, _ = spaces[16]
    qt = np.zeros((2, 2, 2), complex)
    witness = None
    tick = perf_counter()
    for name, pair in pairs.items():
        rho = np.zeros((16, 16), complex)
        rho[:9, :9] = qr @ pair["density"] @ qr.conj().T
        values = np.zeros((3, 2), complex)
        for inp, mu in enumerate([qm[0], qm[1], (qm[0] + qm[1]) / np.sqrt(2)]):
            for output in range(2):
                tag = f"{name}_{inp}_{output}"
                inputs = circuitdir / (tag + ".npz")
                resultpath = circuitdir / (tag + ".json")
                np.savez_compressed(
                    inputs,
                    energies=qe,
                    dipole=mu,
                    output_dipole=qm[output],
                    initial_rho=rho,
                    second_gamma_factor=2.0,
                )
                cmd = [
                    sys.executable,
                    str(root / "scripts/evaluate_response_circuit.py"),
                    str(inputs),
                    str(resultpath),
                    "--photon",
                    str(PHOTON),
                    "--gamma",
                    ".5",
                ]
                if name == "Z" and inp == 0 and output == 0:
                    cmd.append("--hadamard")
                subprocess.run(cmd, check=True)
                result = json.loads(resultpath.read_text())
                values[inp, output] = complex(*result["beta"])
                if "--hadamard" in cmd:
                    witness = result
        local = np.zeros((2, 2, 2), complex)
        local[:, 0, 0] = values[0]
        local[:, 1, 1] = values[1]
        local[:, 0, 1] = local[:, 1, 0] = values[2] - (values[0] + values[1]) / 2
        a = pair["optical_map"]
        qt += np.einsum("ia,jb,kc,abc->ijk", a, a, a, local)
    qseconds = perf_counter() - tick
    qscore = angular_comparison(qt, angles, observed)
    pair_checks = {}
    for name, pair in pairs.items():
        rho = pair["density"]
        pair_checks[name] = {
            "trace": float(np.trace(rho).real),
            "minimum_eigenvalue": float(np.linalg.eigvalsh(rho).min()),
            "purity": float(np.trace(rho @ rho).real),
            "averaged_bonds": pair["bonds"],
            "density": rho,
        }
    reverse = next(
        case["tensor"]
        for case in sensitivities
        if case["name"] == "Opposite handedness"
    )
    reversal_error = float(
        np.linalg.norm(reverse + reference) / np.linalg.norm(reference)
    )
    if reversal_error > 1e-6:
        raise ValueError("Domain reversal failed to reverse the optical amplitude")
    report = {
        "domain_count": 1,
        "physical_parameters_fitted": False,
        "opposite_handedness_amplitude_antisymmetry_error": reversal_error,
        "model": "hybrid: Gao spin state + Fan-model effective optical SHG operator",
        "state_config": asdict(config),
        "gao_config": spin["config"],
        "photon_ev": PHOTON,
        "optical_gamma1_ev": 0.5,
        "optical_gamma2_ev": 1.0,
        "optical_hamiltonian_source": "unchanged Fan reconstructed electronic cluster; NOT Gao spin excitation energies",
        "source_term": "H_Gao - h_chi sum_NN sin(q.delta) n.(Si cross Sj); assumed finite-cluster domain selection, not a published Gao coupling",
        "optical_basis_preprocessing_s": basis_s,
        "spin_state": {
            "energies_ev": state["energies"],
            "weights": state["weights"],
            "thermal_tail_bound": state["thermal_tail_bound"],
            "eigen_residual_max_ev": state["eigen_residual_max_ev"],
            "pair_checks": pair_checks,
        },
        "convergence": rows,
        "quantum_16": {
            "tensor": qt,
            "comparison": qscore,
            "matched_classical_relative_tensor_error": float(
                np.linalg.norm(qt - rows[0]["tensor"])
                / np.linalg.norm(rows[0]["tensor"])
            ),
            "relative_tensor_difference_to_512": float(
                np.linalg.norm(qt - reference) / np.linalg.norm(reference)
            ),
            "evaluation_count": 18,
            "circuit_process_time_s": qseconds,
            "hadamard_witness": witness,
        },
        "classical_512": {"tensor": reference, "comparison": rows[-1]["comparison"]},
        "sensitivities": sensitivities,
        "limitations": [
            "Classical sparse spin diagonalization supplies the density matrix; no quantum thermal-state preparation demonstrated.",
            "Weak chirality source conditions a finite cluster, rather than deriving bulk spontaneous symmetry breaking; results are source and boundary dependent.",
            "Table-S1-caption axes, J2 sign and single-ion factor retain documented Gao source ambiguities.",
            "Fan optical orbital basis and point-center dipoles remain an approximation with missing optical pathways.",
            "16-state optical circuit reduction is not certified converged.",
            "Same held-out folds are used for model diagnostics; post-audit scores are not fresh independent validation.",
            "No calibrated absolute bulk chi or quantum speedup is claimed.",
        ],
        "total_elapsed_s": perf_counter() - start,
    }
    (out / "hybrid_shg_report.json").write_text(json.dumps(serial(report), indent=2))
    np.savez_compressed(
        out / "hybrid_pair_states.npz",
        names=np.array(list(pairs)),
        rho_spin=np.array([p["density"] for p in pairs.values()]),
        optical_maps=np.array([p["optical_map"] for p in pairs.values()]),
        quantum_tensor=qt,
        reference_tensor=reference,
        optical_spin_rotation_16=qr,
        optical_energies_16=qe,
        optical_dipoles_16=np.array(qm),
    )
    grid = np.linspace(0, 360, 721)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for channel, label in enumerate(["Parallel", "Crossed"]):
        ax = axes[channel]
        ax.scatter(
            angles, observed[:, channel], c="black", s=12, label="Gao 1.63 K measured"
        )
        for tensor, score, color, description in [
            (reference, rows[-1]["comparison"], "C0", "Hybrid classical 512"),
            (qt, qscore, "C1", "Hybrid quantum 16"),
        ]:
            fit = score["all_data_fit"]
            ax.plot(
                grid,
                fit["common_scale"]
                * patterns(tensor, grid, fit["offset_deg"])[:, channel],
                c=color,
                label=description,
            )
        ax.set(
            title=label,
            xlabel="Incident polarization angle (deg)",
            ylabel="Signal (uV; common gain fitted)",
        )
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "hybrid_shg_comparison.png", dpi=170)
    plt.close(fig)
    np.savetxt(
        out / "hybrid_shg_predictions.csv",
        np.column_stack(
            [
                angles,
                observed,
                rows[-1]["comparison"]["all_data_prediction_uv"],
                rows[-1]["comparison"]["out_of_fold_prediction_uv"],
                qscore["all_data_prediction_uv"],
            ]
        ),
        delimiter=",",
        header="angle_deg,observed_parallel,observed_crossed,hybrid512_fit_parallel,hybrid512_fit_crossed,hybrid512_heldout_parallel,hybrid512_heldout_crossed,hybrid16_quantum_fit_parallel,hybrid16_quantum_fit_crossed",
        comments="",
    )
    print(
        "Quantum matched error",
        report["quantum_16"]["matched_classical_relative_tensor_error"],
        flush=True,
    )


if __name__ == "__main__":
    main()
