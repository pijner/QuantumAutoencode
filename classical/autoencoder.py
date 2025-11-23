import json
import tensorflow as tf

from time import time
from pathlib import Path
from tensorflow.keras import layers, models


class Autoencoder:
    def __init__(self):
        self.model = self.build_conv_autoencoder()
        self.training_history = None
        self.training_time = None
        self.num_samples_trained_on = None

    def build_conv_autoencoder(self):
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

    def train(self, x_train, x_val=None, epochs=50, batch_size=256, **kwargs):
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

    def reconstruct(self, x):
        return self.model.predict(x)

    def save(self, filepath, description: str = ""):
        # save attributes
        metadata = {
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
    def load(cls, filepath):
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
