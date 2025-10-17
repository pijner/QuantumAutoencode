import matplotlib.pyplot as plt
import tensorflow as tf


def visualize_patches(ds):
    for patches, label in ds.take(2):  # take 2 examples
        patches_np = patches.numpy()
        lbl = label.numpy()
        num_patches = len(patches_np)

        print(f"Label: {lbl} → {num_patches} patches extracted")
        fig, axes = plt.subplots(1, num_patches + 1, figsize=(12, 3))

        # Get the original image (for comparison)
        for img, lbl_orig in ds.take(1):
            orig_image = tf.cast(img, tf.float32) / 255.0
            orig_image = orig_image.numpy().squeeze()

        axes[0].imshow(orig_image, cmap="gray")
        axes[0].set_title("Original 28x28")
        axes[0].axis("off")

        # Display all 8x8 patches
        for i, patch in enumerate(patches_np):
            axes[i + 1].imshow(patch.squeeze(), cmap="gray")
            axes[i + 1].set_title(f"Patch {i + 1}")
            axes[i + 1].axis("off")

        plt.suptitle(f"Label: {lbl} — 8x8 patches (non-overlapping)")
        plt.tight_layout()
        plt.show()
