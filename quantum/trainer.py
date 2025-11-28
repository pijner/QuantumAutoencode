"""Quantum autoencoder trainer module.

This module provides trainer classes for quantum autoencoders with different
dataset loaders for MNIST and Fashion-MNIST datasets.
"""

import logging
import time
from typing import Optional, Tuple, List, Any

import numpy as np
from numpy.typing import NDArray

from classical.utils.data_loader import (
    get_fashion_mnist_datasets,
    get_image_patches_arrays,
    get_mnist_zeros_ones_datasets,
)
from quantum.circuits.qae import QAE


class QAETrainer:
    """Base trainer class for quantum autoencoders.

    This class handles preprocessing, training, prediction, and evaluation
    of quantum autoencoders on image patch data. It automatically calculates
    the required number of qubits based on patch size.

    Parameters
    ----------
    patch_size : int, optional
        Size of square image patches to extract and process, by default 8
    num_trash_qubits : int, optional
        Number of trash qubits to use in the quantum autoencoder, by default 2

    Attributes
    ----------
    patch_size : int
        Size of square image patches
    num_qubits : int
        Total number of qubits needed to encode the patch
    num_trash_qubits : int
        Number of trash qubits
    num_latent_qubits : int
        Number of latent qubits (num_qubits - num_trash_qubits)
    train_set : NDArray or None
        Training dataset
    test_set : NDArray or None
        Test dataset
    qae_model : QAE
        The quantum autoencoder model instance

    Raises
    ------
    AssertionError
        If num_trash_qubits is not less than the total number of qubits
    """

    def __init__(self, patch_size: int = 8, num_trash_qubits: int = 2) -> None:
        self.patch_size = patch_size

        self.num_qubits = (patch_size * patch_size).bit_length() - 1
        assert num_trash_qubits < self.num_qubits, (
            f"Number of trash qubits ({num_trash_qubits}) must be less than total qubits ({self.num_qubits})."
        )
        self.num_trash_qubits = num_trash_qubits
        self.num_latent_qubits = self.num_qubits - self.num_trash_qubits

        self.train_set: Optional[NDArray] = None
        self.test_set: Optional[NDArray] = None

        self.qae_model = QAE(num_latent_qubits=self.num_latent_qubits, num_trash_qubits=self.num_trash_qubits)

    def get_train_test_sets(self) -> Tuple[NDArray, NDArray]:
        """Load and return training and test datasets.

        This method must be implemented by subclasses to provide
        dataset-specific loading logic.

        Returns
        -------
        Tuple[NDArray, NDArray]
            Training and test datasets as numpy arrays

        Raises
        ------
        NotImplementedError
            If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement get_train_test_sets method.")

    def preprocess(
        self,
        keep_unique: bool = True,
        num_train: Optional[int] = None,
        num_test: Optional[int] = None,
        random_seed: int = 42,
    ) -> Tuple[Optional[NDArray], Optional[NDArray]]:
        """Preprocess image data into patches for quantum autoencoder training.

        Extracts image patches, optionally filters for unique patches, normalizes
        to unit vectors, and samples a specified number of training and test samples.

        Parameters
        ----------
        keep_unique : bool, optional
            Whether to keep only unique patches, by default True
        num_train : int or None, optional
            Number of training samples to use. If None, uses all available samples.
            If 0, returns None for training data, by default None
        num_test : int or None, optional
            Number of test samples to use. If None, uses all available samples.
            If 0, returns None for test data, by default None
        random_seed : int, optional
            Random seed for reproducible sampling, by default 42

        Returns
        -------
        Tuple[Optional[NDArray], Optional[NDArray]]
            Preprocessed training and test data as normalized vectors with shape
            (num_samples, patch_size*patch_size), or None if num_train/num_test is 0
        """
        train_ds, test_ds = self.get_train_test_sets()

        train_data = (
            get_image_patches_arrays(train_ds, patch_size=self.patch_size)
            if num_train is None or num_train > 0
            else None
        )
        test_data = (
            get_image_patches_arrays(test_ds, patch_size=self.patch_size) if num_test is None or num_test > 0 else None
        )

        # only keep unique patches
        if keep_unique:
            logging.info("Keeping only unique patches in the dataset.")
            if train_data is not None:
                train_data = np.unique(train_data, axis=0)
            if test_data is not None:
                test_data = np.unique(test_data, axis=0)

        if num_train is not None and train_data is not None:
            if num_train > train_data.shape[0]:
                logging.warning(
                    f"Requested number of training samples ({num_train}) exceeds available unique samples ({train_data.shape[0]}). Using all available samples."
                )
            np.random.seed(random_seed)
            train_indices = np.random.choice(train_data.shape[0], num_train, replace=False)
            train_data = train_data[train_indices]

        if num_test is not None and test_data is not None:
            if num_test > test_data.shape[0]:
                logging.warning(
                    f"Requested number of test samples ({num_test}) exceeds available unique samples ({test_data.shape[0]}). Using all available samples."
                )
            np.random.seed(random_seed)
            test_indices = np.random.choice(test_data.shape[0], num_test, replace=False)
            test_data = test_data[test_indices]

        return train_data, test_data

    def train(
        self,
        initial_params: Optional[List[float]] = None,
        max_iterations: int = 100,
        num_train: Optional[int] = None,
        random_seed: int = 42,
    ) -> Any:
        """Train the quantum autoencoder model on preprocessed image patches.

        Parameters
        ----------
        initial_params : List[float] or None, optional
            Initial parameter values for the variational circuit. If None,
            random parameters are generated, by default None
        max_iterations : int, optional
            Maximum number of optimization iterations, by default 100
        num_train : int or None, optional
            Number of training samples to use. If None, uses all available
            samples, by default None
        random_seed : int, optional
            Random seed for reproducible preprocessing and initialization,
            by default 42

        Returns
        -------
        Any
            Optimization result object containing optimized parameters
        """
        train_data, _ = self.preprocess(num_train=num_train, num_test=0, random_seed=random_seed)

        logging.info(f"Training data shape: {train_data.shape}")

        if initial_params is None:
            np.random.seed(random_seed)
            initial_params = np.random.uniform(0, 2 * np.pi, self.qae_model.model.num_weights).tolist()

        optimized_params = self.qae_model.fit(
            train_images=train_data, initial_param=initial_params, maxiter=max_iterations
        )

        return optimized_params

    def predict(
        self,
        params: Optional[List[float]] = None,
        num_test: Optional[int] = None,
        random_seed: int = 42,
        return_original: bool = True,
    ) -> Tuple[NDArray, NDArray] | NDArray:
        """Run predictions on test dataset.

        Parameters
        ----------
        params : List[float] or None, optional
            Optimized parameters for QAE. If None, uses parameters from training,
            by default None
        num_test : int or None, optional
            Number of test samples to use. If None, uses all available samples,
            by default None
        random_seed : int, optional
            Random seed for reproducibility, by default 42
        return_original : bool, optional
            Whether to return original test data along with predictions, by default True

        Returns
        -------
        Tuple[NDArray, NDArray] or NDArray
            If return_original is True, returns (original_data, reconstructed_data).
            Otherwise, returns only reconstructed_data.
            Shape: (num_samples, patch_size, patch_size)
        """
        _, test_data = self.preprocess(num_train=0, num_test=num_test, random_seed=random_seed)

        logging.info(f"Test data shape: {test_data.shape}")

        t0 = time.time()
        predictions = self.qae_model.predict(test_data, params)
        t1 = time.time()
        logging.info(f"Prediction completed in {t1 - t0:.2f} seconds.")
        predictions = np.reshape(predictions, (-1, self.patch_size, self.patch_size))
        test_data = np.reshape(test_data, (-1, self.patch_size, self.patch_size))

        return test_data, predictions if return_original else predictions

    def evaluate(
        self,
        params: Optional[List[float]] = None,
        num_test: Optional[int] = None,
        random_seed: int = 42,
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """Run predictions and calculate reconstruction metrics.

        Computes reconstruction quality metrics including mean squared error (MSE)
        and fidelity (cosine similarity) between original and reconstructed images.

        Parameters
        ----------
        params : List[float] or None, optional
            Optimized parameters for QAE. If None, uses parameters from training,
            by default None
        num_test : int or None, optional
            Number of test samples to use. If None, uses all available samples,
            by default None
        random_seed : int, optional
            Random seed for reproducibility, by default 42

        Returns
        -------
        Tuple[NDArray, NDArray, NDArray, NDArray]
            A tuple containing:
            - original_data : NDArray
                Original test images, shape (num_samples, patch_size, patch_size)
            - reconstructed_data : NDArray
                Reconstructed images, shape (num_samples, patch_size, patch_size)
            - mse_errors : NDArray
                Mean squared error per sample, shape (num_samples,)
            - fidelities : NDArray
                Fidelity (cosine similarity) per sample, shape (num_samples,)
        """
        original_data, reconstructed_data = self.predict(
            params=params, num_test=num_test, random_seed=random_seed, return_original=True
        )

        mse_errors, fidelities = self.qae_model.calculate_metrics(
            original_data.reshape(-1, self.patch_size * self.patch_size),
            reconstructed_data.reshape(-1, self.patch_size * self.patch_size),
        )

        return original_data, reconstructed_data, mse_errors, fidelities


class MNIST01QAETrainer(QAETrainer):
    """Quantum autoencoder trainer for MNIST digits 0 and 1 only.

    Inherits from QAETrainer and implements dataset loading specifically
    for MNIST images containing only zeros and ones.

    Parameters
    ----------
    patch_size : int, optional
        Size of square image patches to extract and process, by default 8
    num_trash_qubits : int, optional
        Number of trash qubits to use in the quantum autoencoder, by default 2
    """

    def __init__(self, patch_size: int = 8, num_trash_qubits: int = 2) -> None:
        super().__init__(patch_size=patch_size, num_trash_qubits=num_trash_qubits)
        self.train_set: Optional[NDArray] = None
        self.test_set: Optional[NDArray] = None

    def get_train_test_sets(self) -> Tuple[NDArray, NDArray]:
        """Load MNIST zeros and ones datasets.

        Loads and caches the training and test datasets containing only
        MNIST digits 0 and 1.

        Returns
        -------
        Tuple[NDArray, NDArray]
            Training and test datasets containing only digits 0 and 1
        """
        if self.train_set is None or self.test_set is None:
            self.train_set, self.test_set = get_mnist_zeros_ones_datasets()

        return self.train_set, self.test_set


class FashionMNISTQAETrainer(QAETrainer):
    """Quantum autoencoder trainer for Fashion-MNIST dataset.

    Inherits from QAETrainer and implements dataset loading specifically
    for Fashion-MNIST images.

    Parameters
    ----------
    patch_size : int, optional
        Size of square image patches to extract and process, by default 8
    num_trash_qubits : int, optional
        Number of trash qubits to use in the quantum autoencoder, by default 2
    """

    def __init__(self, patch_size: int = 8, num_trash_qubits: int = 2) -> None:
        super().__init__(patch_size=patch_size, num_trash_qubits=num_trash_qubits)
        self.train_set: Optional[NDArray] = None
        self.test_set: Optional[NDArray] = None

    def get_train_test_sets(self) -> Tuple[NDArray, NDArray]:
        """Load Fashion-MNIST datasets.

        Loads and caches the training and test datasets for Fashion-MNIST.

        Returns
        -------
        Tuple[NDArray, NDArray]
            Training and test Fashion-MNIST datasets
        """
        if self.train_set is None or self.test_set is None:
            self.train_set, self.test_set = get_fashion_mnist_datasets()

        return self.train_set, self.test_set
