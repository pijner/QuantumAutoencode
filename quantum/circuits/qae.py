"""Quantum autoencoder implementation using Qiskit.

This module provides a quantum autoencoder (QAE) implementation that uses
quantum circuits to compress and reconstruct quantum states representing
image data.
"""

import json
import logging
import os
import pickle
from datetime import datetime
from time import time
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from tqdm import tqdm


class QAE:
    """Quantum autoencoder for quantum state compression and reconstruction.

    This class implements a quantum autoencoder using a swap test to measure
    the fidelity between encoded and decoded quantum states. The autoencoder
    compresses quantum states by discarding trash qubits and reconstructs
    the original state from the latent representation.

    Parameters
    ----------
    num_latent_qubits : int
        Number of qubits in the latent (compressed) representation
    num_trash_qubits : int
        Number of qubits to discard during compression

    Attributes
    ----------
    num_latent_qubits : int
        Number of qubits in the latent space
    num_trash_qubits : int
        Number of trash qubits
    feature_encoder : RawFeatureVector
        Quantum circuit for encoding classical data into quantum states
    model : SamplerQNN or None
        The quantum neural network model
    autoencoder_circuit : QuantumCircuit or None
        The autoencoder circuit with swap test
    quantum_circuit : QuantumCircuit or None
        Complete quantum circuit combining encoder and autoencoder
    training_history : List[float]
        Loss values recorded during training
    training_time : float
        Total training time in seconds
    optimized_params : Any or None
        Optimized circuit parameters from training
    num_sampled_trained_on : int or None
        Number of samples used in training
    """

    def __init__(self, num_latent_qubits: int, num_trash_qubits: int) -> None:
        self.num_latent_qubits = num_latent_qubits
        self.num_trash_qubits = num_trash_qubits

        self.feature_encoder = RawFeatureVector(2 ** (num_latent_qubits + num_trash_qubits))

        self.model: Optional[SamplerQNN] = None
        self.autoencoder_circuit: Optional[QuantumCircuit] = None
        self.quantum_circuit: Optional[QuantumCircuit] = None

        self.training_history: List[float] = []
        self.training_time: float = 0
        self.optimized_params: Any = None
        self.num_sampled_trained_on: Optional[int] = None

        # initialize circuits
        self.init_circuits()

    def reset_training_params(self) -> None:
        """Reset all training-related parameters to their initial states.

        Clears training history, resets training time to zero, and removes
        optimized parameters. Useful when retraining the model.
        """
        self.training_history = []
        self.training_time = 0
        self.optimized_params = None

    @staticmethod
    def ansatz(num_qubits: int) -> RealAmplitudes:
        """Create a parameterized ansatz circuit.

        Uses the RealAmplitudes variational form with 5 repetitions.

        Parameters
        ----------
        num_qubits : int
            Number of qubits for the ansatz circuit

        Returns
        -------
        RealAmplitudes
            Parameterized quantum circuit ansatz
        """
        return RealAmplitudes(num_qubits, reps=5)

    def init_circuits(self) -> None:
        """Initialize all quantum circuits and the QNN model.

        Creates the autoencoder circuit with swap test, the complete quantum
        circuit, and the SamplerQNN model.
        """
        self.autoencoder_circuit = self._init_autoencoder_circuit()
        self.quantum_circuit = self._init_quantum_circuit()
        self.model = self._init_model()

    def _init_autoencoder_circuit(self) -> QuantumCircuit:
        """Initialize the autoencoder circuit with swap test.

        Creates a quantum circuit that applies a variational ansatz followed
        by a swap test to measure the fidelity between encoded and reference
        trash qubits.

        Returns
        -------
        QuantumCircuit
            Autoencoder circuit with swap test measurement
        """
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

    def _init_quantum_circuit(self) -> QuantumCircuit:
        """Initialize the complete quantum circuit.

        Combines the feature encoder and autoencoder circuit into a single
        quantum circuit for training.

        Returns
        -------
        QuantumCircuit
            Complete quantum circuit with feature encoding and autoencoding
        """
        assert self.autoencoder_circuit is not None, (
            "Autoencoder circuit must be initialized before initializing the quantum circuit."
        )

        self.quantum_circuit = QuantumCircuit(self.num_latent_qubits + 2 * self.num_trash_qubits + 1, 1)
        self.quantum_circuit.compose(
            self.feature_encoder, range(self.num_latent_qubits + self.num_trash_qubits), inplace=True
        )
        self.quantum_circuit = self.quantum_circuit.compose(self.autoencoder_circuit)

        return self.quantum_circuit

    def _init_model(self, output_shape: int = 2, sampler: Any = None) -> SamplerQNN:
        """Initialize the quantum neural network model.

        Creates a SamplerQNN with the quantum circuit, distinguishing between
        input parameters (feature encoding) and weight parameters (trainable).

        Parameters
        ----------
        output_shape : int, optional
            Shape of the output (2 for binary classification), by default 2
        sampler : Any, optional
            Qiskit sampler for circuit execution. If None, uses StatevectorSampler,
            by default None

        Returns
        -------
        SamplerQNN
            Initialized quantum neural network model
        """
        if sampler is None:
            logging.info("No sampler provided, using StatevectorSampler by default.")
            sampler = StatevectorSampler()
        
        assert self.quantum_circuit is not None, (
            "Quantum circuit must be initialized before initializing the model."
        )
        assert self.feature_encoder is not None, (
            "Feature encoder must be initialized before initializing the model."
        )
        assert self.autoencoder_circuit is not None, (
            "Autoencoder circuit must be initialized before initializing the model."
        )

        self.model = SamplerQNN(
            circuit=self.quantum_circuit,
            input_params=self.feature_encoder.parameters,
            weight_params=self.autoencoder_circuit.parameters,
            output_shape=output_shape,
            sampler=sampler,
        )

        return self.model

    def _decode_amplitudes(self, amplitudes: NDArray) -> NDArray:
        """Decode amplitudes from the quantum autoencoder back to normalized image values.

        Parameters
        ----------
        amplitudes : NDArray
            Output from the predict function (probability values from statevector)

        Returns
        -------
        NDArray
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

    def loss_wrapper(self, X: NDArray):
        """Create a loss function for optimization.

        Returns a loss function that computes the average probability of measuring
        the auxiliary qubit in state |1⟩, which corresponds to low fidelity.
        Training minimizes this to maximize reconstruction fidelity.

        Parameters
        ----------
        X : NDArray
            Training data (normalized image vectors)

        Returns
        -------
        callable
            Loss function that takes parameter values and returns a scalar cost
        """

        def loss(params_values: NDArray) -> float:
            probabilities = self.model.forward(X, params_values)
            cost = np.sum(probabilities[:, 1]) / X.shape[0]

            self.training_history.append(cost)

            return cost

        return loss

    def fit(self, train_images: NDArray, initial_param: List[float], maxiter: int = 100) -> Any:
        """Train the quantum autoencoder.

        Optimizes circuit parameters to minimize the loss function (maximize
        reconstruction fidelity) using the COBYLA optimizer.

        Parameters
        ----------
        train_images : NDArray
            Training images as normalized vectors with shape (num_samples, num_features)
        initial_param : List[float]
            Initial parameter values for the variational circuit
        maxiter : int, optional
            Maximum number of optimization iterations, by default 100

        Returns
        -------
        Any
            Optimization result object containing optimized parameters
        """
        # reset training history
        self.reset_training_params()

        optimizer = COBYLA(maxiter=maxiter)

        start_time = time()
        self.optimized_params = optimizer.minimize(fun=self.loss_wrapper(train_images), x0=initial_param)
        end_time = time()

        self.training_time = end_time - start_time
        self.num_sampled_trained_on = train_images.shape[0]

        return self.optimized_params

    def predict(self, X: NDArray, params: Optional[List[float]] = None, return_raw: bool = False) -> NDArray:
        """Predict reconstructed images using the trained quantum autoencoder.

        Parameters
        ----------
        X : NDArray
            Input images (normalized to unit vectors) with shape (num_samples, num_features)
        params : List[float] or None, optional
            Parameters to use for prediction. If None, uses optimized parameters
            from training, by default None
        return_raw : bool, optional
            If True, return raw probability values. If False, return decoded
            amplitudes, by default False

        Returns
        -------
        NDArray
            Reconstructed images as normalized amplitude values (or raw probabilities
            if return_raw=True)
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

    def calculate_metrics(self, original_data: NDArray, predicted_data: NDArray) -> Tuple[NDArray, NDArray]:
        """Calculate reconstruction metrics between original and predicted data.

        Computes mean squared error (MSE) and reconstruction fidelity (cosine
        similarity) between original and predicted images.

        Parameters
        ----------
        original_data : NDArray
            Original input images (normalized to unit vectors)
        predicted_data : NDArray
            Predicted/reconstructed images from the autoencoder

        Returns
        -------
        Tuple[NDArray, NDArray]
            A tuple containing:
            - mse_errors : NDArray
                Mean squared error per sample
            - reconstruction_fidelities : NDArray
                Fidelity (cosine similarity) per sample
        """

        mse_errors = np.mean((original_data - predicted_data) ** 2, axis=1)

        # Calculate fidelity (cosine similarity for normalized vectors)
        reconstruction_fidelities = np.sum(original_data * predicted_data, axis=1)

        return mse_errors, reconstruction_fidelities

    def save_model(self, filepath: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
        """Save the quantum autoencoder model parameters and metadata to file.

        Saves model state to both JSON (for readability) and pickle (for exact
        object reconstruction) formats.

        Parameters
        ----------
        filepath : str
            Path to save the model (without extension)
        metadata : Dict[str, Any] or None, optional
            Additional metadata to save with the model, by default None

        Returns
        -------
        Tuple[str, str]
            Paths to the saved JSON and pickle files

        Raises
        ------
        ValueError
            If the model has not been trained yet
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

    def load_model(self, filepath: str, use_pickle: bool = False) -> Dict[str, Any]:
        """Load quantum autoencoder model parameters from file.

        Parameters
        ----------
        filepath : str
            Path to the saved model (without extension)
        use_pickle : bool, optional
            If True, load from pickle file for exact object reconstruction.
            Otherwise, load from JSON file, by default False

        Returns
        -------
        Dict[str, Any]
            Metadata that was saved with the model

        Raises
        ------
        FileNotFoundError
            If the specified model file is not found
        ValueError
            If model architecture does not match the saved model
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
    def load_from_file(cls, filepath: str, use_pickle: bool = False) -> Tuple["QAE", Dict[str, Any]]:
        """Create a new QAE instance and load model from file.

        Class method that instantiates a QAE with the correct architecture
        from saved model files and loads the trained parameters.

        Parameters
        ----------
        filepath : str
            Path to the saved model (without extension)
        use_pickle : bool, optional
            If True, load from pickle file for exact object reconstruction,
            by default False

        Returns
        -------
        Tuple[QAE, Dict[str, Any]]
            A tuple containing:
            - qae : QAE
                New QAE instance with loaded model parameters
            - metadata : Dict[str, Any]
                Metadata that was saved with the model

        Raises
        ------
        FileNotFoundError
            If the specified model file is not found
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

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the current model state.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing model information including:
            - architecture : dict
                Qubit counts and circuit parameters
            - training_status : dict
                Training state and loss history
            - optimizer_result : dict
                Optimization results (if trained)
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
    def list_saved_models(directory: str = ".") -> List[str]:
        """List all saved QAE models in a directory.

        Searches for valid QAE model JSON files in the specified directory
        and returns their base paths.

        Parameters
        ----------
        directory : str, optional
            Directory to search for saved models, by default "."

        Returns
        -------
        List[str]
            Sorted list of model file paths (without extensions)
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
