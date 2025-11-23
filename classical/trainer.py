import logging
import numpy as np

from time import time

from classical.utils.data_loader import (
    get_mnist_zeros_ones_datasets,
    get_fashion_mnist_datasets,
    get_image_patches_arrays,
)
from classical.autoencoder import Autoencoder


class AutoencoderTrainer:
    def __init__(self, patch_size: int = 8):
        self.patch_size = patch_size
        self.train_set = None
        self.test_set = None
        self.autoencoder_model = Autoencoder()

    def get_train_test_sets(self):
        raise NotImplementedError("Subclasses must implement get_train_test_sets method.")

    def preprocess(self, keep_unique: bool = True, num_train: int = None, num_test: int = None, random_seed: int = 42):
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

    def train(self, num_train, random_seed=42, epochs=200, **kwargs):
        x_train, _ = self.preprocess(keep_unique=True, num_train=num_train, num_test=0, random_seed=random_seed)

        self.autoencoder_model.train(x_train, x_val=None, epochs=epochs, batch_size=256, **kwargs)

    def predict(self, num_test: int = None, random_seed: int = 42, return_original: bool = True):
        _, x_test = self.preprocess(keep_unique=True, num_train=0, num_test=num_test, random_seed=random_seed)
        start_time = time()
        predictions = self.autoencoder_model.reconstruct(x_test)
        end_time = time()
        logging.info(f"Prediction time for {x_test.shape[0]} samples: {end_time - start_time:.4f} seconds")
        return x_test, predictions if return_original else predictions

    def evaluate(self, num_test: int = None, random_seed: int = 42):
        original, reconstructed = self.predict(num_test=num_test, random_seed=random_seed, return_original=True)
        # Calculate MSE per sample
        mse = np.mean((original - reconstructed) ** 2, axis=(1, 2, 3))

        # Calculate fidelity  (cosine similarity for normalized vectors)
        norm_x_test = original / np.linalg.norm(original, axis=(1, 2), keepdims=True)
        norm_reconstructed = reconstructed / np.linalg.norm(reconstructed, axis=(1, 2), keepdims=True)
        fidelity = np.sum(norm_x_test * norm_reconstructed, axis=(1, 2, 3))

        return original, reconstructed, mse, fidelity


class MNIST01AutoencoderTrainer(AutoencoderTrainer):
    def __init__(self, patch_size: int = 8):
        super().__init__(patch_size=patch_size)

    def get_train_test_sets(self):
        if self.train_set is None or self.test_set is None:
            self.train_set, self.test_set = get_mnist_zeros_ones_datasets()

        return self.train_set, self.test_set


class FashionMNISTAutoencoderTrainer(AutoencoderTrainer):
    def __init__(self, patch_size: int = 8):
        super().__init__(patch_size=patch_size)

    def get_train_test_sets(self):
        if self.train_set is None or self.test_set is None:
            self.train_set, self.test_set = get_fashion_mnist_datasets()

        return self.train_set, self.test_set
