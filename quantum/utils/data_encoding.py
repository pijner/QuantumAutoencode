"""Quantum data encoding utilities.

This module provides functions for encoding classical data into quantum states
using amplitude encoding, which maps classical data to quantum amplitudes.
"""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation


def amplitude_encode(data: NDArray, num_qubits: int) -> Tuple[NDArray, int]:
    """Encode classical data into a quantum state using amplitude encoding.

    Normalizes the input data to unit vector and converts it to complex
    amplitudes suitable for quantum state preparation. Handles zero-norm
    inputs by returning the |0...0⟩ state.

    Parameters
    ----------
    data : NDArray
        The input classical data to be encoded (1D array)
    num_qubits : int
        The number of qubits to use for encoding

    Returns
    -------
    Tuple[NDArray, int]
        A tuple containing:
        - state : NDArray
            Encoded quantum state as complex amplitudes
        - flag : int
            Flag indicating whether input was all zeros (1) or not (0)

    Raises
    ------
    ValueError
        If the input data is not a 1D array or if its length exceeds the
        maximum length for the given number of qubits
    """
    if data.ndim != 1:
        raise ValueError("Input data must be a 1D array.")

    max_data_length = 2**num_qubits

    if len(data) > max_data_length:
        raise ValueError(f"Data length {len(data)} exceeds maximum length {max_data_length} for {num_qubits} qubits.")

    if len(data) < max_data_length:
        data = np.pad(data, (0, max_data_length - len(data)), "constant")

    norm = np.linalg.norm(data)

    if norm == 0:
        # If the norm is zero, return the |0...0> state and a flag indicating zero input
        state = np.zeros(data.shape[0], dtype=complex)
        state[0] = 1.0
        flag = 1
    else:
        state = (data / norm).astype(complex)
        flag = 0
    return state, flag


def prepare_amplitude_encoding_circuit(data: NDArray, num_qubits: int) -> QuantumCircuit:
    """Prepare a quantum circuit for amplitude encoding of classical data.

    Creates a quantum circuit that encodes classical data into quantum amplitudes
    using StatePreparation. An additional flag qubit indicates whether the input
    was all zeros.

    The last qubit acts as a flag qubit:
      - |1⟩ if the input data was all zeros
      - |0⟩ otherwise

    Parameters
    ----------
    data : NDArray
        Input data in 1D array format
    num_qubits : int
        Number of qubits to use for encoding (excluding the flag qubit)

    Returns
    -------
    QuantumCircuit
        A quantum circuit with num_qubits + 1 qubits for amplitude encoding
        of the input data
    """
    state, flag = amplitude_encode(data, num_qubits)
    qc = QuantumCircuit(num_qubits + 1, name="AmplitudeEncode")

    state_prep = StatePreparation(state)
    qc.append(state_prep, qc.qubits[:-1])

    # Set the lsb to |1> if input data was all zeros
    if flag == 1:
        qc.x(qc.qubits[0])

    return qc
