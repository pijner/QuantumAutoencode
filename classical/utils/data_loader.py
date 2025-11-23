import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm


def load_mnist_datasets():
    (ds_train, ds_test), ds_info = tfds.load(
        "mnist", split=["train", "test"], shuffle_files=True, as_supervised=True, with_info=True
    )

    return ds_train, ds_test, ds_info


def filter_zeros_ones(image, label):
    return tf.logical_or(tf.equal(label, 0), tf.equal(label, 1))


def get_mnist_zeros_ones_datasets():
    ds_train, ds_test, _ = load_mnist_datasets()

    ds_train_01 = ds_train.filter(filter_zeros_ones)
    ds_test_01 = ds_test.filter(filter_zeros_ones)

    return ds_train_01, ds_test_01


def get_fashion_mnist_datasets():
    (ds_train, ds_test) = tfds.load(
        "fashion_mnist", split=["train", "test"], shuffle_files=True, as_supervised=True, with_info=False
    )

    return ds_train, ds_test


def preprocess_and_extract_patches(image, patch_size: int = 8):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, axis=0)

    patches_flat = tf.image.extract_patches(
        images=image,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )

    patches_flat = tf.squeeze(patches_flat, axis=0)
    num_y = tf.shape(patches_flat)[0]
    num_x = tf.shape(patches_flat)[1]

    # Reshape flat 64-length patches into (8,8,1)
    patches = tf.reshape(patches_flat, (num_y, num_x, patch_size, patch_size))
    patches = tf.reshape(patches, (-1, patch_size, patch_size, 1))

    return patches


def reconstruct_from_patches(patches: np.ndarray, image_height: int, image_width: int, patch_size=8):
    """
    patches: numpy array of shape (num_patches, patch_size, patch_size, 1)
    image_height, image_width: original image dimensions
    """
    num_y = image_height // patch_size
    num_x = image_width // patch_size

    # reshape patches back to grid
    grid = patches.reshape(num_y, num_x, patch_size, patch_size, 1)

    # initialize output image
    reconstructed = np.zeros((image_height, image_width, 1), dtype=patches.dtype)

    # fill the image block-by-block
    for i in range(num_y):
        for j in range(num_x):
            reconstructed[i * patch_size : (i + 1) * patch_size, j * patch_size : (j + 1) * patch_size, :] = grid[i, j]

    return reconstructed


def get_image_patches_arrays(dataset: tf.data.Dataset, patch_size: int = 8, dropna: bool = True) -> np.ndarray:
    patched_dataset = []
    for image, _ in tqdm(dataset):
        patches = preprocess_and_extract_patches(image, patch_size=patch_size)
        patched_dataset.append(patches.numpy().reshape(-1, (patch_size * patch_size)))

    # reshape to (num_images * num_patches, patch_size*patch_size)
    patched_dataset = np.array(patched_dataset, dtype=np.float64).reshape(-1, patch_size * patch_size)

    # normalize patches by dividing by sum of squares (per patch)
    norms = np.linalg.norm(patched_dataset, axis=1, keepdims=True)
    patched_dataset = patched_dataset / norms

    # drop any NaN patches resulting from zero division
    if dropna:
        patched_dataset = patched_dataset[~np.isnan(patched_dataset).any(axis=1)]

    return patched_dataset
