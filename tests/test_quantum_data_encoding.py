import numpy as np

from qiskit.quantum_info import Statevector
from quantum.utils.data_encoding import amplitude_encode, prepare_amplitude_encoding_circuit


def test_amplitude_encode_nonzero():
    data = np.array([1, 2, 3, 4])
    state, flag = amplitude_encode(data, num_qubits=2)

    expected_norm = np.linalg.norm(data)
    np.testing.assert_allclose(state, data / expected_norm)
    assert flag == 0


def test_amplitude_encode_zero_vector():
    data = np.zeros(4)
    state, flag = amplitude_encode(data, num_qubits=2)
    expected = np.zeros(4, dtype=complex)
    expected[0] = 1.0
    np.testing.assert_allclose(state, expected)
    assert flag == 1


def test_amplitude_encode_padding():
    data = np.array([1, 2])
    state, flag = amplitude_encode(data, num_qubits=2)
    assert len(state) == 4
    assert np.isclose(np.linalg.norm(state), 1.0)
    assert flag == 0


def test_prepare_amplitude_encoding_circuit_nonzero():
    data = np.array([1, 2, 3, 4])
    qc = prepare_amplitude_encoding_circuit(data, num_qubits=2)

    sv = Statevector.from_instruction(qc)
    probs = np.abs(sv.data) ** 2

    # Last (flag) qubit should remain |0>
    assert np.isclose(np.sum(probs[:4]), 1.0)
    assert np.allclose(probs[4:], 0.0)


def test_prepare_amplitude_encoding_circuit_zero_input():
    data = np.zeros(4)
    qc = prepare_amplitude_encoding_circuit(data, num_qubits=2)

    sv = Statevector.from_instruction(qc)
    probs = np.abs(sv.data) ** 2

    # Expect |00>|1> → index 1 in statevector
    expected = np.zeros(8)
    expected[1] = 1.0

    np.testing.assert_allclose(probs, expected)
