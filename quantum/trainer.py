import logging
import numpy as np

from classical.utils.data_loader import get_image_patches_arrays, get_mnist_zeros_ones_datasets
from quantum.circuits.qae import QAE


class MNIST01QAETrainer:
    def __init__(self, patch_size: int = 8, num_trash_qubits: int = 2):
        self.patch_size = patch_size

        self.num_qubits = (patch_size * patch_size).bit_length() - 1
        assert num_trash_qubits < self.num_qubits, (
            f"Number of trash qubits ({num_trash_qubits}) must be less than total qubits ({self.num_qubits})."
        )
        self.num_trash_qubits = num_trash_qubits
        self.num_latent_qubits = self.num_qubits - self.num_trash_qubits

        self.qae_model = QAE(num_latent_qubits=self.num_latent_qubits, num_trash_qubits=self.num_trash_qubits)

    def preprocess(self, keep_unique: bool = True, num_train: int = None, num_test: int = None, random_seed: int = 42):
        mnist_01_train, mnist_01_test = get_mnist_zeros_ones_datasets()

        train_data = get_image_patches_arrays(mnist_01_train, patch_size=self.patch_size)
        test_data = get_image_patches_arrays(mnist_01_test, patch_size=self.patch_size)

        # only keep unique patches
        if keep_unique:
            logging.info("Keeping only unique patches in the dataset.")
            logging.info(f"Original train data shape: {train_data.shape}, test data shape: {test_data.shape}")
            train_data = np.unique(train_data, axis=0)
            test_data = np.unique(test_data, axis=0)
            logging.info(f"Unique train data shape: {train_data.shape}, test data shape: {test_data.shape}")

        if num_train is not None:
            np.random.seed(random_seed)
            train_indices = np.random.choice(train_data.shape[0], num_train, replace=False)
            train_data = train_data[train_indices]

        if num_test is not None:
            np.random.seed(random_seed)
            test_indices = np.random.choice(test_data.shape[0], num_test, replace=False)
            test_data = test_data[test_indices]

        return train_data, test_data

    def train(
        self,
        initial_params: list[float] = None,
        max_iterations: int = 100,
        num_train: int = None,
        random_seed: int = 42,
    ):
        train_data, _ = self.preprocess(num_train=num_train, random_seed=random_seed)

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
        params: list[float] = None,
        num_test: int = None,
        random_seed: int = 42,
        return_original: bool = True,
    ):
        _, test_data = self.preprocess(num_test=num_test, random_seed=random_seed)

        logging.info(f"Test data shape: {test_data.shape}")

        predictions = self.qae_model.predict(test_data, params)
        predictions = np.reshape(predictions, (-1, self.patch_size, self.patch_size))

        return test_data, predictions if return_original else predictions
