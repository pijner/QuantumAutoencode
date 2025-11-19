import numpy as np
import logging

from time import time

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector

from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from tqdm import tqdm


class QAE:
    def __init__(self, num_latent_qubits: int, num_trash_qubits: int):
        self.num_latent_qubits = num_latent_qubits
        self.num_trash_qubits = num_trash_qubits

        self.feature_encoder = RawFeatureVector(2 ** (num_latent_qubits + num_trash_qubits))

        self.model = None
        self.autoencoder_circuit = None
        self.quantum_circuit = None

        self.training_history = []
        self.training_time = 0
        self.optimized_params = None

        # initialize circuits
        self.init_circuits()

    def reset_training_params(self):
        self.training_history = []
        self.training_time = 0
        self.optimized_params = None

    @staticmethod
    def ansatz(num_qubits):
        return RealAmplitudes(num_qubits, reps=5)

    def init_circuits(self):
        self.autoencoder_circuit = self._init_autoencoder_circuit()
        self.quantum_circuit = self._init_quantum_circuit()
        self.model = self._init_model()

    def _init_autoencoder_circuit(self):
        self.autoencoder_circuit = QuantumCircuit(
            QuantumRegister(self.num_latent_qubits + 2 * self.num_trash_qubits + 1, "q"),
            ClassicalRegister(1, "c"),
        )

        self.autoencoder_circuit.compose(
            self.ansatz(self.num_latent_qubits + self.num_trash_qubits),
            range(0, self.num_latent_qubits + self.num_trash_qubits),
            inplace=True,
        )

        self.autoencoder_circuit.barrier()
        auxiliary_qubit = self.num_latent_qubits + 2 * self.num_trash_qubits

        # swap test
        self.autoencoder_circuit.h(auxiliary_qubit)
        for i in range(self.num_trash_qubits):
            self.autoencoder_circuit.cswap(
                auxiliary_qubit, self.num_latent_qubits + i, self.num_latent_qubits + self.num_trash_qubits + i
            )

        self.autoencoder_circuit.h(auxiliary_qubit)
        self.autoencoder_circuit.measure(auxiliary_qubit, self.autoencoder_circuit.clbits[0])

        return self.autoencoder_circuit

    def _init_quantum_circuit(self):
        self.quantum_circuit = QuantumCircuit(self.num_latent_qubits + 2 * self.num_trash_qubits + 1, 1)
        self.quantum_circuit = self.quantum_circuit.compose(
            self.feature_encoder, range(self.num_latent_qubits + self.num_trash_qubits)
        )
        self.quantum_circuit = self.quantum_circuit.compose(self.autoencoder_circuit)

        return self.quantum_circuit

    def _init_model(self, output_shape: int = 2, sampler=None):
        if sampler is None:
            logging.info("No sampler provided, using StatevectorSampler by default.")
            sampler = StatevectorSampler()

        self.model = SamplerQNN(
            circuit=self.quantum_circuit,
            input_params=self.feature_encoder.parameters,
            weight_params=self.autoencoder_circuit.parameters,
            output_shape=output_shape,
            sampler=sampler,
        )

        return self.model

    def loss_wrapper(self, X: np.ndarray):
        def loss(params_values):
            probabilities = self.model.forward(X, params_values)
            cost = np.sum(probabilities[:, 1]) / X.shape[0]

            self.training_history.append(cost)

            return cost

        return loss

    def fit(self, train_images: np.ndarray, initial_param: list[float], maxiter: int = 100):
        # reset training history
        self.reset_training_params()

        optimizer = COBYLA(maxiter=maxiter)

        start_time = time()
        self.optimized_params = optimizer.minimize(fun=self.loss_wrapper(train_images), x0=initial_param)
        end_time = time()

        self.training_time = end_time - start_time

        return self.optimized_params

    def predict(self, X: np.ndarray, params: list[float] = None):
        if params is None:
            params = self.optimized_params.x
        else:
            assert len(params) == self.model.num_weights, (
                f"Number of params give ({len(params)}) do not match expected ({self.model.num_weights})"
            )

        predict_circuit = QuantumCircuit(self.num_latent_qubits + self.num_trash_qubits)
        predict_circuit = predict_circuit.compose(self.feature_encoder)

        ansatz = self.ansatz(self.num_latent_qubits + self.num_trash_qubits)
        predict_circuit = predict_circuit.compose(ansatz)

        predict_circuit.barrier()

        # reset trash qubits
        for i in range(self.num_latent_qubits, self.num_latent_qubits + self.num_trash_qubits):
            predict_circuit.reset(i)

        predict_circuit.barrier()
        predict_circuit = predict_circuit.compose(ansatz.inverse())

        predict_values = []

        for image in tqdm(X):
            param_values = np.concatenate((image, self.optimized_params.x))
            output_qc = predict_circuit.assign_parameters(param_values)
            predict_values.append(Statevector(output_qc).data)

        return np.array(predict_values).real ** 2
