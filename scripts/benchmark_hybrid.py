"""Matched 18-component CPU benchmark for the current hybrid only.

Inputs preloaded; both solvers evaluated in the same process. Preprocessing,
input I/O and hardware sampling are excluded. No external runtime is inferred.
"""

import argparse
import json
import math
import platform
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np

from nii2_shg.material_response import classical_response
from nii2_shg.response_circuit import quantum_response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root
    report = json.loads((root / "outputs/hybrid_shg_report.json").read_text())
    inputs = []
    for path in sorted((root / "outputs/hybrid_circuits").glob("*.npz")):
        with np.load(path) as saved:
            inputs.append({k: saved[k].copy() for k in saved.files})
    if len(inputs) != 18:
        raise ValueError("Expected 18 current-model optical circuit inputs")
    photon = report["photon_ev"]
    gamma = report["optical_gamma1_ev"]

    def evaluate(quantum):
        values = []
        for case in inputs:
            kwargs = {
                "output_mu": case["output_dipole"],
                "initial_rho": case["initial_rho"],
                "second_gamma_factor": float(case["second_gamma_factor"]),
            }
            if quantum:
                result = quantum_response(
                    case["energies"],
                    case["dipole"],
                    photon,
                    gamma,
                    hadamard=False,
                    **kwargs,
                )
                values.append(result["beta"])
            else:
                values.append(
                    classical_response(
                        case["energies"], case["dipole"], photon, gamma, **kwargs
                    )
                )
        return np.array(values)

    # One warm-up per path; fixed repetitions are recorded separately.
    classical = evaluate(False)
    quantum = evaluate(True)
    ct = []
    qt = []
    for _ in range(31):
        tick = perf_counter()
        classical = evaluate(False)
        ct.append(perf_counter() - tick)
    for _ in range(3):
        tick = perf_counter()
        quantum = evaluate(True)
        qt.append(perf_counter() - tick)
    witness = report["quantum_16"]["hadamard_witness"]
    beta = complex(*witness["beta"])
    scale = witness["scale"]
    a = beta / scale
    relative_rms = 0.1
    shots = math.ceil(scale**2 * (2 - abs(a) ** 2) / (relative_rms**2 * abs(beta) ** 2))
    output = {
        "scope": "same-process, same 18 current-model scalar optical responses, preloaded inputs",
        "classical_repeats": 31,
        "quantum_repeats": 3,
        "classical_median_s": float(np.median(ct)),
        "classical_min_s": min(ct),
        "classical_max_s": max(ct),
        "quantum_statevector_median_s": float(np.median(qt)),
        "quantum_min_s": min(qt),
        "quantum_max_s": max(qt),
        "statevector_to_classical_time_ratio": float(np.median(qt) / np.median(ct)),
        "matched_relative_l2_error": float(
            np.linalg.norm(quantum - classical) / np.linalg.norm(classical)
        ),
        "matched_max_absolute_error": float(np.max(abs(quantum - classical))),
        "count": 18,
        "electronic_states": 16,
        "system_qubits": 8,
        "work_qubits": 4,
        "hadamard_control_qubits": 1,
        "sampling_included": False,
        "preprocessing_included": False,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "qiskit": version("qiskit"),
        "settings": "single-thread BLAS, ideal CPU Statevector; dense matrix/gate/state construction included; no Hadamard circuit in timed path",
        "optical_basis_preprocessing_s_separate": report[
            "optical_basis_preprocessing_s"
        ],
        "sampling_estimate": {
            "case": "Z_0_0",
            "normalization": scale,
            "beta_real": beta.real,
            "beta_imag": beta.imag,
            "normalized_overlap_magnitude": abs(a),
            "target_complex_relative_rms": relative_rms,
            "shots_per_quadrature": shots,
            "total_shots_two_quadratures": 2 * shots,
            "meaning": "independent +/-1 Hadamard samples, RMS criterion from exact variance; not an optimized or fundamental hardware resource bound; no amplitude estimation/amplification implemented",
        },
        "advantage_conclusion": "No computational quantum advantage demonstrated. Potential asymptotic benefits require efficient many-body oracles/state preparation and feasible readout, which this dense small-instance implementation does not provide.",
    }
    destination = args.output or root / "outputs/hybrid_benchmark.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2))
    print(
        json.dumps(
            {
                k: output[k]
                for k in [
                    "classical_median_s",
                    "quantum_statevector_median_s",
                    "statevector_to_classical_time_ratio",
                    "matched_relative_l2_error",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
