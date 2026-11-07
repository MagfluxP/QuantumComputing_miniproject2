"""Evaluate one saved finite response circuit in a fresh process."""

import argparse
import json
from pathlib import Path

import numpy as np

from nii2_shg.response_circuit import quantum_response

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs")
    parser.add_argument("output")
    parser.add_argument("--photon", type=float, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--hadamard", action="store_true")
    args = parser.parse_args()
    with np.load(args.inputs) as saved:
        e, m = saved["energies"], saved["dipole"]
        output_mu = saved["output_dipole"] if "output_dipole" in saved.files else None
        initial_rho = saved["initial_rho"] if "initial_rho" in saved.files else None
        second_gamma_factor = (
            float(saved["second_gamma_factor"])
            if "second_gamma_factor" in saved.files
            else 1.0
        )
    result = quantum_response(
        e,
        m,
        args.photon,
        args.gamma,
        args.hadamard,
        output_mu=output_mu,
        initial_rho=initial_rho,
        second_gamma_factor=second_gamma_factor,
    )
    report = {
        key: value
        for key, value in result.items()
        if key not in ("circuit", "beta", "hadamard_beta")
    }
    report["beta"] = [result["beta"].real, result["beta"].imag]
    report["hadamard_beta"] = (
        None
        if result["hadamard_beta"] is None
        else [result["hadamard_beta"].real, result["hadamard_beta"].imag]
    )
    report["circuit_text"] = str(result["circuit"].draw(output="text", fold=110))
    Path(args.output).write_text(json.dumps(report))
