"""Classical autoencoder trainer module.

This module provides trainer classes for classical convolutional autoencoders
with different dataset loaders for MNIST and Fashion-MNIST datasets.
"""

import logging
from time import time
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from classical.utils.data_loader import (
    get_mnist_zeros_ones_datasets,
    get_fashion_mnist_datasets,
    get_image_patches_arrays,
)
from classical.autoencoder import Autoencoder


class AutoencoderTrainer:
    """Base trainer class for classical convolutional autoencoders.

    This class handles preprocessing, training, prediction, and evaluation
    of classical autoencoders on image patch data.

    Parameters
    ----------
    patch_size : int, optional
        Size of square image patches to extract and process, by default 8

    Attributes
    ----------
    patch_size : int
        Size of square image patches
    train_set : NDArray or None
        Training dataset
    test_set : NDArray or None
        Test dataset
    autoencoder_model : Autoencoder
        The classical autoencoder model instance
    """

    def __init__(self, patch_size: int = 8) -> None:
        self.patch_size = patch_size
        self.train_set: Optional[NDArray] = None
        self.test_set: Optional[NDArray] = None
        self.autoencoder_model = Autoencoder()

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
        """Preprocess image data into patches for autoencoder training.

        Extracts image patches, optionally filters for unique patches,
        and samples a specified number of training and test samples.

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
            Preprocessed training and test data with shape
            (num_samples, patch_size, patch_size, 1), or None if num_train/num_test is 0
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
            if train_data is not None:
                train_data = np.unique(train_data, axis=0)
            if test_data is not None:
                test_data = np.unique(test_data, axis=0)

        # get_image_patches_arrays reshapes to (num_images * num_patches, patch_size*patch_size)
        # we need (num_samples, patch_size, patch_size, 1)
        if train_data is not None:
            train_data = train_data.reshape(-1, self.patch_size, self.patch_size, 1)
        if test_data is not None:
            test_data = test_data.reshape(-1, self.patch_size, self.patch_size, 1)

        if num_train is not None and num_train > 0:
            if num_train > train_data.shape[0]:
                logging.warning(
                    f"Requested number of training samples ({num_train}) exceeds available unique samples ({train_data.shape[0]}). Using all available samples."
                )
            np.random.seed(random_seed)
            train_indices = np.random.choice(train_data.shape[0], num_train, replace=False)
            train_data = train_data[train_indices]

        if num_test is not None and num_test > 0:
            if num_test > test_data.shape[0]:
                logging.warning(
                    f"Requested number of test samples ({num_test}) exceeds available unique samples ({test_data.shape[0]}). Using all available samples."
                )
            np.random.seed(random_seed)
            test_indices = np.random.choice(test_data.shape[0], num_test, replace=False)
            test_data = test_data[test_indices]

        return train_data, test_data

    def train(self, num_train: int, random_seed: int = 42, epochs: int = 200, **kwargs) -> None:
        """Train the autoencoder model on preprocessed image patches.

        Parameters
        ----------
        num_train : int
            Number of training samples to use
        random_seed : int, optional
            Random seed for reproducible preprocessing, by default 42
        epochs : int, optional
            Number of training epochs, by default 200
        **kwargs
            Additional keyword arguments passed to the autoencoder's train method
        """
        x_train, _ = self.preprocess(keep_unique=True, num_train=num_train, num_test=0, random_seed=random_seed)

        self.autoencoder_model.train(x_train, x_val=None, epochs=epochs, batch_size=256, **kwargs)

    def predict(
        self,
        num_test: Optional[int] = None,
        random_seed: int = 42,
        return_original: bool = True,
    ) -> Tuple[NDArray, NDArray] | NDArray:
        """Generate predictions using the trained autoencoder.

        Parameters
        ----------
        num_test : int or None, optional
            Number of test samples to predict on. If None, uses all available samples,
            by default None
        random_seed : int, optional
            Random seed for reproducible preprocessing, by default 42
        return_original : bool, optional
            Whether to return both original and reconstructed images, by default True

        Returns
        -------
        Tuple[NDArray, NDArray] or NDArray
            If return_original is True, returns (original_images, reconstructed_images).
            Otherwise, returns only reconstructed_images.
            Shape: (num_samples, patch_size, patch_size, 1)
        """
        _, x_test = self.preprocess(keep_unique=True, num_train=0, num_test=num_test, random_seed=random_seed)
        start_time = time()
        predictions = self.autoencoder_model.reconstruct(x_test)
        end_time = time()
        logging.info(f"Prediction time for {x_test.shape[0]} samples: {end_time - start_time:.4f} seconds")
        return x_test, predictions if return_original else predictions

    def evaluate(
        self,
        num_test: Optional[int] = None,
        random_seed: int = 42,
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """Evaluate autoencoder performance on test data.

        Computes reconstruction quality metrics including mean squared error (MSE)
        and fidelity (cosine similarity) between original and reconstructed images.

        Parameters
        ----------
        num_test : int or None, optional
            Number of test samples to evaluate. If None, uses all available samples,
            by default None
        random_seed : int, optional
            Random seed for reproducible preprocessing, by default 42

        Returns
        -------
        Tuple[NDArray, NDArray, NDArray, NDArray]
            A tuple containing:
            - original : NDArray
                Original test images, shape (num_samples, patch_size, patch_size, 1)
            - reconstructed : NDArray
                Reconstructed images, shape (num_samples, patch_size, patch_size, 1)
            - mse : NDArray
                Mean squared error per sample, shape (num_samples,)
            - fidelity : NDArray
                Fidelity (cosine similarity) per sample, shape (num_samples,)
        """
        original, reconstructed = self.predict(num_test=num_test, random_seed=random_seed, return_original=True)
        # Calculate MSE per sample
        mse = np.mean((original - reconstructed) ** 2, axis=(1, 2, 3))

        # Calculate fidelity  (cosine similarity for normalized vectors)
        norm_x_test = original / np.linalg.norm(original, axis=(1, 2), keepdims=True)
        norm_reconstructed = reconstructed / np.linalg.norm(reconstructed, axis=(1, 2), keepdims=True)
        fidelity = np.sum(norm_x_test * norm_reconstructed, axis=(1, 2, 3))

        return original, reconstructed, mse, fidelity


class MNIST01AutoencoderTrainer(AutoencoderTrainer):
    """Autoencoder trainer for MNIST digits 0 and 1 only.

    Inherits from AutoencoderTrainer and implements dataset loading
    specifically for MNIST images containing only zeros and ones.

    Parameters
    ----------
    patch_size : int, optional
        Size of square image patches to extract and process, by default 8
    """

    def __init__(self, patch_size: int = 8) -> None:
        super().__init__(patch_size=patch_size)

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


class FashionMNISTAutoencoderTrainer(AutoencoderTrainer):
    """Autoencoder trainer for Fashion-MNIST dataset.

    Inherits from AutoencoderTrainer and implements dataset loading
    specifically for Fashion-MNIST images.

    Parameters
    ----------
    patch_size : int, optional
        Size of square image patches to extract and process, by default 8
    """

    def __init__(self, patch_size: int = 8) -> None:
        super().__init__(patch_size=patch_size)

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
