import numpy as np
import logging
import pickle
import json
import os

from time import time
from datetime import datetime

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
        self.num_sampled_trained_on = None

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

    def _decode_amplitudes(self, amplitudes: np.ndarray) -> np.ndarray:
        """
        Decode amplitudes from the quantum autoencoder back to normalized image values.

        Args:
            amplitudes: Output from the predict function (probability values from statevector)

        Returns:
            Normalized image values ready for comparison with original inputs
        """
        # Convert probabilities back to amplitudes (square root)
        decoded_amplitudes = np.sqrt(np.abs(amplitudes))

        # For batch processing, handle both single images and batches
        if decoded_amplitudes.ndim == 1:
            # Single image
            # Extract only the relevant portion (first 2^(num_latent + num_trash) elements)
            num_features = 2 ** (self.num_latent_qubits + self.num_trash_qubits)
            decoded_image = decoded_amplitudes[:num_features]

            # Normalize to unit vector (L2 norm = 1) to match original preprocessing
            norm = np.sqrt(np.sum(decoded_image**2))
            if norm > 0:
                decoded_image = decoded_image / norm

            return decoded_image
        else:
            # Batch of images
            num_features = 2 ** (self.num_latent_qubits + self.num_trash_qubits)
            batch_size = decoded_amplitudes.shape[0]
            decoded_images = np.zeros((batch_size, num_features))

            for i in range(batch_size):
                # Extract relevant portion for each image
                decoded_image = decoded_amplitudes[i, :num_features]

                # Normalize each image to unit vector
                norm = np.sqrt(np.sum(decoded_image**2))
                if norm > 0:
                    decoded_images[i] = decoded_image / norm
                else:
                    decoded_images[i] = decoded_image

            return decoded_images

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
        self.num_sampled_trained_on = train_images.shape[0]

        return self.optimized_params

    def predict(self, X: np.ndarray, params: list[float] = None, return_raw: bool = False):
        """
        Predict reconstructed images using the trained quantum autoencoder.

        Args:
            X: Input images (normalized to unit vectors)
            params: Optional parameters to use (defaults to optimized parameters from training)
            return_raw: If True, return raw probability values. If False, return decoded amplitudes.

        Returns:
            Reconstructed images as normalized amplitude values (or raw probabilities if return_raw=True)
        """
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

        raw_probabilities = np.array(predict_values).real ** 2

        if return_raw:
            return raw_probabilities
        else:
            return self._decode_amplitudes(raw_probabilities)

    def calculate_metrics(self, original_data: np.ndarray, predicted_data: np.ndarray):
        """
        Reconstruct images and calculate reconstruction metrics.

        Args:
            X: Input images (normalized to unit vectors)
            params: Optional parameters to use (defaults to optimized parameters from training)

        Returns:
            tuple: (reconstructed_images, mse_errors, reconstruction_fidelities)
        """

        mse_errors = np.mean((original_data - predicted_data) ** 2, axis=1)

        # Calculate fidelity (cosine similarity for normalized vectors)
        reconstruction_fidelities = np.sum(original_data * predicted_data, axis=1)

        return mse_errors, reconstruction_fidelities

    def save_model(self, filepath: str, metadata: dict = None):
        """
        Save the quantum autoencoder model parameters and metadata to file.

        Args:
            filepath: Path to save the model (without extension)
            metadata: Optional dictionary with additional metadata to save
        """
        if self.optimized_params is None:
            raise ValueError("Model has not been trained yet. Cannot save untrained model.")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)

        # Prepare model state dictionary
        model_state = {
            "num_latent_qubits": self.num_latent_qubits,
            "num_trash_qubits": self.num_trash_qubits,
            "optimized_params": self.optimized_params.x.tolist(),
            "training_history": self.training_history,
            "training_time": self.training_time,
            "optimizer_result": {
                "fun": getattr(self.optimized_params, "fun", None),
                "nfev": getattr(self.optimized_params, "nfev", None),
                "success": getattr(self.optimized_params, "success", True),  # Default to True if not available
                "message": getattr(self.optimized_params, "message", "Optimization completed"),
            },
            "saved_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        # Save as JSON for readability and compatibility
        json_filepath = f"{filepath}.json"
        with open(json_filepath, "w") as f:
            json.dump(model_state, f, indent=2)

        # Also save as pickle for exact object reconstruction if needed
        pickle_filepath = f"{filepath}.pkl"
        with open(pickle_filepath, "wb") as f:
            pickle.dump({"model_state": model_state, "optimized_params_object": self.optimized_params}, f)

        logging.info(f"Model saved to {json_filepath} and {pickle_filepath}")
        return json_filepath, pickle_filepath

    def load_model(self, filepath: str, use_pickle: bool = False):
        """
        Load quantum autoencoder model parameters from file.

        Args:
            filepath: Path to the saved model (without extension)
            use_pickle: If True, load from pickle file for exact object reconstruction

        Returns:
            dict: Metadata that was saved with the model
        """
        if use_pickle:
            pickle_filepath = f"{filepath}.pkl"
            if not os.path.exists(pickle_filepath):
                raise FileNotFoundError(f"Pickle file not found: {pickle_filepath}")

            with open(pickle_filepath, "rb") as f:
                data = pickle.load(f)
                model_state = data["model_state"]
                self.optimized_params = data["optimized_params_object"]
        else:
            json_filepath = f"{filepath}.json"
            if not os.path.exists(json_filepath):
                raise FileNotFoundError(f"JSON file not found: {json_filepath}")

            with open(json_filepath, "r") as f:
                model_state = json.load(f)

            # Reconstruct optimized_params as a simple object with necessary attributes
            class OptimizedParams:
                def __init__(self, x, fun, nfev, success, message):
                    self.x = np.array(x)
                    self.fun = fun
                    self.nfev = nfev
                    self.success = success
                    self.message = message

            opt_result = model_state["optimizer_result"]
            self.optimized_params = OptimizedParams(
                model_state["optimized_params"],
                opt_result.get("fun", None),
                opt_result.get("nfev", None),
                opt_result.get("success", True),
                opt_result.get("message", "Optimization completed"),
            )

        # Verify model architecture matches
        if (
            model_state["num_latent_qubits"] != self.num_latent_qubits
            or model_state["num_trash_qubits"] != self.num_trash_qubits
        ):
            raise ValueError(
                f"Model architecture mismatch. "
                f"Expected: {self.num_latent_qubits} latent, {self.num_trash_qubits} trash qubits. "
                f"Found: {model_state['num_latent_qubits']} latent, {model_state['num_trash_qubits']} trash qubits."
            )

        # Load training history and metadata
        self.training_history = model_state["training_history"]
        self.training_time = model_state["training_time"]

        logging.info(
            f"Model loaded successfully. "
            f"Training history: {len(self.training_history)} iterations, "
            f"Final loss: {self.training_history[-1]:.6f}"
        )

        return model_state.get("metadata", {})

    @classmethod
    def load_from_file(cls, filepath: str, use_pickle: bool = False):
        """
        Class method to create a new QAE instance and load model from file.

        Args:
            filepath: Path to the saved model (without extension)
            use_pickle: If True, load from pickle file for exact object reconstruction

        Returns:
            QAE: New QAE instance with loaded model parameters
        """
        # First load the model state to get architecture parameters
        if use_pickle:
            pickle_filepath = f"{filepath}.pkl"
            if not os.path.exists(pickle_filepath):
                raise FileNotFoundError(f"Pickle file not found: {pickle_filepath}")

            with open(pickle_filepath, "rb") as f:
                data = pickle.load(f)
                model_state = data["model_state"]
        else:
            json_filepath = f"{filepath}.json"
            if not os.path.exists(json_filepath):
                raise FileNotFoundError(f"JSON file not found: {json_filepath}")

            with open(json_filepath, "r") as f:
                model_state = json.load(f)

        # Create new QAE instance with correct architecture
        qae = cls(num_latent_qubits=model_state["num_latent_qubits"], num_trash_qubits=model_state["num_trash_qubits"])

        # Load the model parameters
        metadata = qae.load_model(filepath, use_pickle=use_pickle)

        return qae, metadata

    def get_model_info(self):
        """
        Get comprehensive information about the current model state.

        Returns:
            dict: Model information including architecture, training status, and performance
        """
        info = {
            "architecture": {
                "num_latent_qubits": self.num_latent_qubits,
                "num_trash_qubits": self.num_trash_qubits,
                "total_qubits": self.num_latent_qubits + 2 * self.num_trash_qubits + 1,
                "num_parameters": len(self.autoencoder_circuit.parameters) if self.autoencoder_circuit else 0,
            },
            "training_status": {
                "is_trained": self.optimized_params is not None,
                "training_iterations": len(self.training_history),
                "training_time": self.training_time,
                "final_loss": self.training_history[-1] if self.training_history else None,
                "initial_loss": self.training_history[0] if self.training_history else None,
            },
        }

        if self.optimized_params is not None:
            info["optimizer_result"] = {
                "final_cost": getattr(self.optimized_params, "fun", None),
                "function_evaluations": getattr(self.optimized_params, "nfev", None),
                "optimization_success": getattr(self.optimized_params, "success", True),
                "optimizer_message": getattr(self.optimized_params, "message", "Optimization completed"),
            }

        return info

    @staticmethod
    def list_saved_models(directory: str = "."):
        """
        List all saved QAE models in a directory.

        Args:
            directory: Directory to search for saved models

        Returns:
            list: List of model file paths (without extensions)
        """
        if not os.path.exists(directory):
            return []

        json_files = []
        for file in os.listdir(directory):
            if file.endswith(".json"):
                # Check if it's a valid QAE model file
                base_name = file[:-5]  # Remove .json extension
                json_path = os.path.join(directory, file)

                # Verify it's a QAE model by checking the structure
                try:
                    with open(json_path, "r") as f:
                        data = json.load(f)
                        if all(key in data for key in ["num_latent_qubits", "num_trash_qubits", "optimized_params"]):
                            json_files.append(os.path.join(directory, base_name))
                except (json.JSONDecodeError, KeyError):
                    continue

        return sorted(json_files)
