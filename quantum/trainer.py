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

        self.train_set = None
        self.test_set = None

        self.qae_model = QAE(num_latent_qubits=self.num_latent_qubits, num_trash_qubits=self.num_trash_qubits)

    def get_train_test_sets(self):
        if self.train_set is None or self.test_set is None:
            self.train_set, self.test_set = get_mnist_zeros_ones_datasets()

        return self.train_set, self.test_set

    def preprocess(self, keep_unique: bool = True, num_train: int = None, num_test: int = None, random_seed: int = 42):
        mnist_01_train, mnist_01_test = self.get_train_test_sets()

        train_data = (
            get_image_patches_arrays(mnist_01_train, patch_size=self.patch_size)
            if num_train is None or num_train > 0
            else None
        )
        test_data = (
            get_image_patches_arrays(mnist_01_test, patch_size=self.patch_size)
            if num_test is None or num_test > 0
            else None
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
        initial_params: list[float] = None,
        max_iterations: int = 100,
        num_train: int = None,
        random_seed: int = 42,
    ):
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
        self, params: list[float] = None, num_test: int = None, random_seed: int = 42, return_original: bool = True
    ):
        """
        Runs predictions on test dataset

        Parameters
        ----------
        params : list[float], optional
            optimized parameters for QAE, by default None
        num_test : int, optional
            number of test samples to use, by default None
        random_seed : int, optional
            random seed for reproducibility, by default 42
        return_original : bool, optional
            whether to return original test data along with predictions, by default True

        Returns
        -------
        tuple or np.ndarray
            If return_original is True, returns a tuple (original_data, reconstructed_data).
            Otherwise, returns only the reconstructed data.
        """
        _, test_data = self.preprocess(num_train=0, num_test=num_test, random_seed=random_seed)

        logging.info(f"Test data shape: {test_data.shape}")

        predictions = self.qae_model.predict(test_data, params)
        predictions = np.reshape(predictions, (-1, self.patch_size, self.patch_size))
        test_data = np.reshape(test_data, (-1, self.patch_size, self.patch_size))

        return test_data, predictions if return_original else predictions

    def evaluate(self, params: list[float] = None, num_test: int = None, random_seed: int = 42):
        """
        Runs predictions and calculates metrics

        Parameters
        ----------
        params : list[float], optional
            optimized parameters for QAE, by default None
        num_test : int, optional
            number of test samples to use, by default None
        random_seed : int, optional
            random seed for reproducibility, by default 42

        Returns
        -------
        tuple
            A tuple containing original data, reconstructed data, mean squared errors, and fidelities.
        """
        original_data, reconstructed_data = self.predict(
            params=params, num_test=num_test, random_seed=random_seed, return_original=True
        )

        mse_errors, fidelities = self.qae_model.calculate_metrics(
            original_data.reshape(-1, self.patch_size * self.patch_size),
            reconstructed_data.reshape(-1, self.patch_size * self.patch_size),
        )

        return original_data, reconstructed_data, mse_errors, fidelities
