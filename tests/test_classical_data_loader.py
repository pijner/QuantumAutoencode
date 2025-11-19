import tensorflow as tf

from classical.utils.data_loader import preprocess_and_extract_patches

def test_patch_extraction():
    patch_size = 8
    image = tf.reshape(tf.range(28*28, dtype=tf.float32), (28, 28, 1))    

    patches = preprocess_and_extract_patches(image)

    expected_num_patches = (28 // patch_size) ** 2  # 3×3 = 9
    tf.debugging.assert_equal(tf.shape(patches)[0], expected_num_patches)
    tf.debugging.assert_equal(tf.shape(patches)[1:], [patch_size, patch_size, 1])

    # Check top-left patch matches original top-left
    top_left_patch = tf.squeeze(patches[0])
    expected_patch = tf.squeeze(image[:patch_size, :patch_size, :] / 255.0)
    tf.debugging.assert_near(top_left_patch, expected_patch, atol=1e-6)

    print("✅ Patch extraction test passed (9 non-overlapping 8x8 patches)")