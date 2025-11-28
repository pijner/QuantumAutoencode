"""Classical convolutional autoencoder implementation.

This module provides a convolutional autoencoder using TensorFlow/Keras
for image compression and reconstruction tasks.
"""

import json
from pathlib import Path
from time import time
from typing import Optional, Dict, Any, List

import tensorflow as tf
from numpy.typing import NDArray
from tensorflow.keras import layers, models


class Autoencoder:
    """Convolutional autoencoder for image compression and reconstruction.

    This class implements a convolutional autoencoder with an encoder that
    compresses 8x8 image patches into a 32-dimensional latent space and a
    decoder that reconstructs the original images.

    Attributes
    ----------
    model : tf.keras.Model
        The complete autoencoder model (encoder + decoder)
    training_history : List[float] or None
        Loss values recorded during training
    training_time : float or None
        Total time taken for training in seconds
    num_samples_trained_on : int or None
        Number of samples used in the most recent training
    _encoder : tf.keras.Model
        The encoder portion of the autoencoder
    _decoder : tf.keras.Model
        The decoder portion of the autoencoder
    """

    def __init__(self) -> None:
        self.model: tf.keras.Model = self.build_conv_autoencoder()
        self.training_history: Optional[List[float]] = None
        self.training_time: Optional[float] = None
        self.num_samples_trained_on: Optional[int] = None

    def build_conv_autoencoder(self) -> tf.keras.Model:
        """Build and compile the convolutional autoencoder architecture.

        Constructs an encoder that compresses 8x8x1 images into a 32-dimensional
        latent space through convolutional and pooling layers, and a decoder that
        reconstructs the original images using upsampling and transposed convolutions.

        Returns
        -------
        tf.keras.Model
            Compiled autoencoder model with Adam optimizer and MSE loss
        """
        # Encoder
        self._encoder = models.Sequential(
            [
                layers.Input(shape=(8, 8, 1)),
                layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
                layers.MaxPooling2D((2, 2), padding="same"),
                layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
                layers.MaxPooling2D((2, 2), padding="same"),
                layers.Flatten(),  # Flatten the output
                layers.Dense(32),  # Latent space with 32 units
            ]
        )

        # Decoder
        self._decoder = models.Sequential(
            [
                layers.InputLayer(input_shape=(32,)),  # Input shape of the latent vector
                layers.Dense(2 * 2 * 32, activation="relu"),  # Dense layer to reshape latent space
                layers.Reshape((2, 2, 32)),  # Reshape back to a suitable shape for the decoder
                layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
                layers.UpSampling2D((2, 2)),
                layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
                layers.UpSampling2D((2, 2)),
                layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same"),
            ]
        )

        # Autoencoder combining encoder and decoder
        self.model = models.Sequential([self._encoder, self._decoder])
        self.model.compile(optimizer="adam", loss="mse")
        return self.model

    def train(
        self,
        x_train: NDArray,
        x_val: Optional[NDArray] = None,
        epochs: int = 50,
        batch_size: int = 256,
        **kwargs: Any,
    ) -> None:
        """Train the autoencoder model.

        Trains the autoencoder to reconstruct input images by minimizing
        mean squared error between input and output. Records training history,
        training time, and number of samples.

        Parameters
        ----------
        x_train : NDArray
            Training data with shape (num_samples, height, width, channels)
        x_val : NDArray or None, optional
            Validation data with same shape as x_train. If provided, validation
            loss will be computed during training, by default None
        epochs : int, optional
            Number of training epochs, by default 50
        batch_size : int, optional
            Batch size for training, by default 256
        **kwargs : Any
            Additional keyword arguments passed to model.fit()
        """
        start_time = time()
        training_history = self.model.fit(
            x_train,
            x_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(x_val, x_val) if x_val is not None else None,
            **kwargs,
        )

        self.training_history = training_history.history["loss"]

        end_time = time()
        self.training_time = end_time - start_time
        self.num_samples_trained_on = x_train.shape[0]

    def reconstruct(self, x: NDArray) -> NDArray:
        """Reconstruct images using the trained autoencoder.

        Passes input images through the encoder-decoder pipeline to generate
        reconstructed images.

        Parameters
        ----------
        x : NDArray
            Input images with shape (num_samples, height, width, channels)

        Returns
        -------
        NDArray
            Reconstructed images with the same shape as input
        """
        return self.model.predict(x)

    def save(self, filepath: str, description: str = "") -> None:
        """Save the autoencoder model and metadata to disk.

        Saves the Keras model to a .keras file and metadata (training history,
        training time, number of samples, description) to a .json file.

        Parameters
        ----------
        filepath : str
            Base path for saving files. Extensions will be automatically added:
            .keras for the model and .json for metadata
        description : str, optional
            Optional description of the model to include in metadata, by default ""
        """
        # save attributes
        metadata: Dict[str, Any] = {
            "training_history": self.training_history if self.training_history else None,
            "training_time": self.training_time,
            "num_samples_trained_on": self.num_samples_trained_on,
            "description": description,
        }

        # save metadata to json file alongside model
        with open(Path(filepath).with_suffix(".json"), "w") as f:
            json.dump(metadata, f)

        # strip file extension for tf.keras save
        filepath_keras = str(Path(filepath).with_suffix(".keras"))
        self.model.save(filepath_keras)

    @classmethod
    def load(cls, filepath: str) -> "Autoencoder":
        """Load a saved autoencoder model from disk.

        Loads both the Keras model from a .keras file and metadata from a .json
        file, restoring the complete autoencoder state.

        Parameters
        ----------
        filepath : str
            Base path to the saved files (without extension). Will look for
            .keras and .json files with this base name

        Returns
        -------
        Autoencoder
            Loaded autoencoder instance with model weights and metadata restored
        """
        autoencoder = cls()
        filepath_keras = str(Path(filepath).with_suffix(".keras"))
        autoencoder.model = tf.keras.models.load_model(filepath_keras)

        # load attributes
        metadata_path = Path(filepath).with_suffix(".json")
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            if metadata:
                if metadata.get("training_history"):
                    autoencoder.training_history = tf.keras.callbacks.History()
                    autoencoder.training_history.history = metadata["training_history"]
                autoencoder.training_time = metadata.get("training_time")
            autoencoder.num_samples_trained_on = metadata.get("num_samples_trained_on")

        return autoencoder
