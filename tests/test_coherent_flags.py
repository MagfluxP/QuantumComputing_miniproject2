"""Complex accepted products retain fresh ancilla flags rather than recycling failure."""

import numpy as np
from numpy.testing import assert_allclose
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from nii2_shg.circuits import lcu, overlap_circuit


def test_complex_product_for_all_inputs_and_failure_flag_isolation():
    # Two four-dimensional, noncommuting contractions; each needs one LCU flag.
    identity = np.eye(4)
    x = np.array([[0, 1], [1, 0]])
    z = np.diag([1, -1])
    a = lcu([0.6, 0.4], [identity, np.kron(x, np.eye(2))])
    b = lcu([0.7, 0.3j], [identity, np.kron(z, x)])
    product = overlap_circuit([a, b])
    assert product.num_qubits == 4
    # Compare the whole accepted operator, not only a single basis input.
    assert_allclose(
        Operator(product).data[:4, :4], b.accepted() @ a.accepted(), atol=2e-12
    )
    first = QuantumCircuit(4)
    first.append(a.circuit.to_gate(), [0, 1, 2])
    failed = Statevector.from_instruction(first).data.copy()
    failed[(np.arange(16) & (1 << 2)) == 0] = 0
    assert np.linalg.norm(failed) > 0.1
    second = QuantumCircuit(4)
    second.append(b.circuit.to_gate(), [0, 1, 3])
    # A rejected first flag cannot return to all-zero flags in later operations.
    final = Statevector(failed).evolve(second).data
    assert_allclose(final[:4], 0, atol=2e-12)
