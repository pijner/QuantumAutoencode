import tensorflow as tf
import tensorflow_datasets as tfds


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


def preprocess_and_extract_patches(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # normalize
    image = tf.expand_dims(image, axis=0)  # (1,28,28,1)

    patch_size = 8
    patches_flat = tf.image.extract_patches(
        images=image,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )  # shape (1, 3, 3, 64)

    patches_flat = tf.squeeze(patches_flat, axis=0)  # remove batch
    num_y = tf.shape(patches_flat)[0]
    num_x = tf.shape(patches_flat)[1]

    # Reshape flat 64-length patches into (8,8,1)
    patches = tf.reshape(patches_flat, (num_y, num_x, patch_size, patch_size))
    patches = tf.reshape(patches, (-1, patch_size, patch_size, 1))  # 9 patches total

    return patches, label
