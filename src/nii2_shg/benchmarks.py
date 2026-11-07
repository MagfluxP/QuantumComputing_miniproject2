"""Matched-model accuracy and cost measurements; no invented literature timing."""

import platform
import statistics
import time
from importlib.metadata import version

import numpy as np
from qiskit.quantum_info import Statevector

from .circuits import overlap_circuit, sample_hadamard
from .model import OpticalModel, beta_from_products, classical_beta, photon_energy
from .qsp import design_window, qsp_sequence


def run_benchmark(degrees: tuple[int, ...] = (4, 8, 16, 32), repeats: int = 5) -> dict:
    """Return measured timings/errors for identical model, frequency and damping.

    All wall times are local classical CPU times including quantum simulation.
    Phase design is measured cold per degree; circuit construction is separate.
    No hardware wall time or comparison against the authors' execution is inferred.
    """
    if repeats < 1 or not degrees:
        raise ValueError("repeats must be positive and degrees nonempty")
    model = OpticalModel()
    energy = photon_energy(1030)
    gamma = 0.05
    times = []
    for _ in range(101):
        start = time.perf_counter()
        reference = classical_beta(model, energy, gamma)
        times.append(time.perf_counter() - start)
    rows = []
    last = None
    for degree in degrees:
        design_window.cache_clear()
        start = time.perf_counter()
        design = design_window(degree)
        phase_seconds = time.perf_counter() - start
        start = time.perf_counter()
        blocks = qsp_sequence(model, degree=degree)
        circuit = overlap_circuit(blocks)
        construction_seconds = time.perf_counter() - start
        scale = float(np.prod([b.scale for b in blocks]))
        simulation_times = []
        for _ in range(repeats):
            start = time.perf_counter()
            product = complex(Statevector.from_instruction(circuit).data[0] * scale)
            simulation_times.append(time.perf_counter() - start)
        # Hermitian reversed product is its conjugate; known zero terms omitted.
        products = {(2, 1): product, (1, 2): product.conjugate()}
        start = time.perf_counter()
        beta = beta_from_products(model, products, energy, gamma)
        assembly_seconds = time.perf_counter() - start
        rows.append(
            {
                "degree": degree,
                "phase_synthesis_s": phase_seconds,
                "circuit_construction_s": construction_seconds,
                "statevector_median_s": statistics.median(simulation_times),
                "statevector_min_s": min(simulation_times),
                "statevector_max_s": max(simulation_times),
                "response_assembly_s": assembly_seconds,
                "phase_error_grid": design.synthesis_error,
                "smooth_window_error_grid": design.window_error,
                "product_real": product.real,
                "product_imag": product.imag,
                "product_absolute_error": abs(product - 0.12),
                "beta_real": beta.real,
                "beta_imag": beta.imag,
                "beta_absolute_error": abs(beta - reference),
                "beta_relative_error": abs(beta - reference) / abs(reference),
                "qubits_product": circuit.num_qubits,
                "qubits_hadamard": circuit.num_qubits + 1,
                "total_normalization": scale,
                "logical_signal_calls_unoptimized_realpart": 4 * degree,
                "shots_exact_statevector": 0,
            }
        )
        last = product, scale
    rng = np.random.default_rng(7)
    sampling = []
    for shots in (1000, 10000, 100000, 1000000):
        errors = [
            abs(sample_hadamard(last[0], last[1], shots, rng) - last[0]) ** 2
            for _ in range(400)
        ]
        sampling.append(
            {
                "shots_per_quadrature": shots,
                "total_shots": 2 * shots,
                "rms_statistical_error": float(np.sqrt(np.mean(errors))),
            }
        )
    return {
        "model": "illustrative three-level optical model, not NiI2",
        "wavelength_nm": 1030,
        "gamma_ev": gamma,
        "reference_beta_real": reference.real,
        "reference_beta_imag": reference.imag,
        "classical_reference_median_s": statistics.median(times),
        "classical_reference_repeats": 101,
        "simulation_repeats": repeats,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "qiskit": version("qiskit"),
        "settings": "CPU Statevector, ideal/no noise; no hardware backend",
        "rows": rows,
        "sampling": sampling,
        "literature_accuracy": None,
        "experimental_accuracy": None,
        "literature_runtime_s": None,
        "quantum_hardware_runtime_s": None,
        "unavailable_reason": "No validated matched material prediction, complex theoretical response or external timing logs",
    }
