"""Small-instance gate/resource counts with explicit compilation and success scope."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

from .circuits import dipole_block, overlap_circuit
from .model import OpticalModel
from .qsp import qsp_filter, qsp_sequence, qsp_signal_circuit


def collect_resources(
    degrees: tuple[int, ...] = (4, 8, 16, 32), seed_transpiler: int = 7
) -> dict[str, Any]:
    """Return logical queries, compiled toy counts and postselection probabilities.

    Inputs are even QSP degrees 2..64 and a fixed compiler seed. Counts use
    basis u/cx, all-to-all connectivity and optimization level 1. Dense filter
    resynthesis is separate from repeated-signal query costs. No hardware run.
    """
    model = OpticalModel()
    rows = []

    def compile_counts(circuit: QuantumCircuit) -> dict[str, Any]:
        tick = perf_counter()
        compiled = transpile(
            circuit,
            basis_gates=["u", "cx"],
            coupling_map=None,
            optimization_level=1,
            seed_transpiler=seed_transpiler,
        )
        return {
            "qubits": compiled.num_qubits,
            "depth": compiled.depth(),
            "cx": compiled.count_ops().get("cx", 0),
            "u": compiled.count_ops().get("u", 0),
            "compile_s": perf_counter() - tick,
        }

    dipole = compile_counts(dipole_block(model).circuit)
    for degree in degrees:
        signal, _ = qsp_signal_circuit(model, 1.5, degree)
        first, _ = qsp_filter(model, 1.5, degree)
        second, _ = qsp_filter(model, 2.7, degree)
        raw = compile_counts(signal)
        f1 = compile_counts(first.circuit)
        f2 = compile_counts(second.circuit)
        blocks = qsp_sequence(model, degree=degree)
        circuit = overlap_circuit(blocks)
        state = Statevector.from_instruction(circuit).data
        success = float(np.sum(abs(state[:4]) ** 2))
        rows.append(
            {
                "degree": degree,
                "raw_sequence_signal_calls": degree,
                "two_filters_unoptimized_controlled_signal_calls": 4 * degree,
                "raw_qsp": raw,
                "dense_realpart_filter_1": f1,
                "dense_realpart_filter_2": f2,
                "base_product_component_sum_cx": 3 * dipole["cx"] + f1["cx"] + f2["cx"],
                "base_product_component_sum_u": 3 * dipole["u"] + f1["u"] + f2["u"],
                "product_system_qubits": 2,
                "product_work_qubits": sum(b.ancillas for b in blocks),
                "product_total_qubits": circuit.num_qubits,
                "hadamard_total_qubits": circuit.num_qubits + 1,
                "accepted_work_zero_probability": success,
                "expected_attempts_if_postselecting_state": 1 / success,
                "product_normalization": float(np.prod([b.scale for b in blocks])),
            }
        )
    return {
        "basis_gates": ["u", "cx"],
        "connectivity": "all-to-all; coupling_map=None",
        "optimization_level": 1,
        "seed_transpiler": seed_transpiler,
        "simulator": "qiskit.quantum_info.Statevector, ideal CPU",
        "executed_shots": 0,
        "dipole": dipole,
        "rows": rows,
        "compilation_scope": "raw three-qubit QSP sequence and dense small dipole/filter components, including PREP/SELECT/UNPREP; component sums are not a compiled full controlled-Hadamard circuit",
        "success_scope": "probability sums over all four system outputs with all work flags zero; expected attempts applies only to postselected state generation, not an extra factor multiplying Hadamard shot estimates",
        "oracle_scope": "small diagonal signal matrix and dense matrices constructed classically, not scalable chemistry oracles",
        "material_logical_resources": {
            "electronic_states": 16,
            "liouville_system_qubits": 8,
            "fresh_work_qubits": 4,
            "hadamard_control_qubits": 1,
            "base_operations": [
                "density loading",
                "C",
                "R1",
                "C",
                "R2",
                "dipole readout",
            ],
            "physical_basis_synthesis": "not attempted for large dense material blocks; no physical material gate count claimed",
            "all_18_evaluations": "18 base overlaps; one additional real+imag ideal Hadamard witness in saved calculator; no hardware samples",
        },
    }
