"""
MNIST 0/1 Quantum Autoencoder Interactive Demo
Interactive Gradio interface to visualize quantum autoencoder results on MNIST digits.
"""

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import logging
import os
import json
from datetime import datetime

from quantum.trainer import MNIST01QAETrainer
from quantum.circuits.qae import QAE

sns.set_theme(style="whitegrid")
logging.basicConfig(level=logging.INFO)


class QAEDemo:
    def __init__(self):
        self.trainer = None
        self.originals = None
        self.predictions = None
        self.mse_errors = None
        self.fidelities = None
        self.training_time = 0
        self.inference_time = 0
        self.is_trained = False

    def train_model(self, num_train, max_iterations, random_seed):
        """Train the quantum autoencoder model"""
        try:
            # Initialize trainer
            self.trainer = MNIST01QAETrainer()

            # Train the model
            start_time = time.time()
            self.trainer.train(num_train=num_train, max_iterations=max_iterations, random_seed=random_seed)
            self.training_time = time.time() - start_time

            self.is_trained = True

            # Generate training history plot
            training_plot = self.create_training_plot()

            return (
                f"Training completed successfully!\n\n"
                f"Training time: {self.training_time:.2f} seconds\n\n"
                f"Final loss: {self.trainer.qae_model.training_history[-1]:.6f}\n\n"
                f"Number of iterations: {len(self.trainer.qae_model.training_history)}",
                training_plot,
            )

        except Exception as e:
            return f"Training failed: {str(e)}", None

    def evaluate_model(self, num_test, random_seed):
        """Evaluate the model on test data"""
        if not self.is_trained or self.trainer is None:
            return "Please train the model first!", None, None, None, None, None

        try:
            # Run evaluation
            start_time = time.time()
            self.originals, self.predictions, self.mse_errors, self.fidelities = self.trainer.evaluate(
                num_test=num_test, random_seed=random_seed
            )
            self.inference_time = time.time() - start_time

            # Calculate summary statistics
            avg_mse = np.mean(self.mse_errors)
            avg_fidelity = np.mean(self.fidelities)

            # Create metrics plot
            metrics_plot = self.create_metrics_plot()

            # Create initial comparison (first image)
            comparison_plot = self.create_comparison_plot(0)

            return (
                f"Evaluation completed!\n\n"
                f"Inference time: {self.inference_time:.2f} seconds\n\n"
                f"Average MSE: {avg_mse:.6f}\n\n"
                f"Average Fidelity: {avg_fidelity:.6f}\n\n"
                f"Number of test samples: {len(self.originals)}",
                metrics_plot,
                comparison_plot,
                gr.update(minimum=0, maximum=len(self.originals) - 1, value=0, visible=True, interactive=True),
                f"Image 1/{len(self.originals)}",
                self.get_image_info(0),
            )

        except Exception as e:
            return f"Evaluation failed: {str(e)}", None, None, None, None, None

    def create_training_plot(self):
        """Create training history visualization"""
        if self.trainer is None or not hasattr(self.trainer.qae_model, "training_history"):
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        history = self.trainer.qae_model.training_history
        iterations = range(1, len(history) + 1)

        sns.lineplot(x=iterations, y=history, ax=ax)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title(f"Training Loss History - Final Loss: {history[-1]:.6f}")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def create_metrics_plot(self):
        """Create metrics distribution plots"""
        if self.mse_errors is None or self.fidelities is None:
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # MSE distribution
        sns.histplot(self.mse_errors, bins=20, kde=True, color="red", alpha=0.7, ax=ax1)
        ax1.set_xlabel("MSE")
        ax1.set_ylabel("Frequency")
        ax1.set_title(f"MSE Distribution\nMean: {np.mean(self.mse_errors):.6f}")

        # Fidelity distribution
        sns.histplot(self.fidelities, bins=20, kde=True, color="blue", alpha=0.7, ax=ax2)
        ax2.set_xlabel("Fidelity")
        ax2.set_ylabel("Frequency")
        ax2.set_title(f"Fidelity Distribution\nMean: {np.mean(self.fidelities):.6f}")

        plt.tight_layout()
        return fig

    def create_comparison_plot(self, image_idx):
        """Create side-by-side comparison of original and reconstructed image"""
        if self.originals is None or self.predictions is None or image_idx >= len(self.originals):
            return None

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

        original = self.originals[image_idx]
        reconstructed = self.predictions[image_idx]
        difference = np.abs(original - reconstructed)

        # Original image
        sns.heatmap(original, cmap="gray", cbar=True, ax=ax1, cbar_kws={"shrink": 0.8})
        ax1.set_title("Original", fontsize=12, fontweight="bold")
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Reconstructed image
        sns.heatmap(reconstructed, cmap="gray", cbar=True, ax=ax2, cbar_kws={"shrink": 0.8})
        ax2.set_title("Reconstructed", fontsize=12, fontweight="bold")
        ax2.set_xticks([])
        ax2.set_yticks([])

        # Difference
        sns.heatmap(difference, cmap="Reds", cbar=True, ax=ax3, cbar_kws={"shrink": 0.8})
        ax3.set_title("Absolute Difference", fontsize=12, fontweight="bold")
        ax3.set_xticks([])
        ax3.set_yticks([])

        plt.tight_layout()
        return fig

    def get_image_info(self, image_idx):
        """Get detailed information about a specific image"""
        if self.mse_errors is None or self.fidelities is None or image_idx >= len(self.mse_errors):
            return "No data available"

        return (
            f"**Image {image_idx + 1} Metrics:**\n\n"
            f"• MSE: {self.mse_errors[image_idx]:.6f}\n\n"
            f"• Fidelity: {self.fidelities[image_idx]:.6f}\n\n"
            f"• Original norm: {np.linalg.norm(self.originals[image_idx]):.6f}\n\n"
            f"• Reconstructed norm: {np.linalg.norm(self.predictions[image_idx]):.6f}"
        )

    def update_image_display(self, image_idx):
        """Update the image comparison when slider changes"""
        if self.originals is None or self.predictions is None:
            return None, "No images available", "No data available"

        # Ensure image_idx is within bounds
        image_idx = max(0, min(int(image_idx), len(self.originals) - 1))

        comparison_plot = self.create_comparison_plot(image_idx)
        image_counter = f"Image {image_idx + 1}/{len(self.originals)}"
        image_info = self.get_image_info(image_idx)

        return comparison_plot, image_counter, image_info

    def save_model(self, model_name, description):
        """Save the trained model to file"""
        if not self.is_trained or self.trainer is None:
            return "Please train the model first!", gr.update(choices=[])

        if not model_name.strip():
            return "Please provide a valid model name!", gr.update(choices=[])

        try:
            # Create models directory if it doesn't exist
            models_dir = "saved_models"
            os.makedirs(models_dir, exist_ok=True)

            # Create filename with timestamp to avoid conflicts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name}_{timestamp}"
            filepath = os.path.join(models_dir, filename)

            # Prepare metadata
            metadata = {
                "description": description,
                "num_sampled_trained_on": self.trainer.qae_model.num_sampled_trained_on
                if hasattr(self.trainer.qae_model, "num_sampled_trained_on")
                else "Unknown",
                "test_samples": len(self.originals) if self.originals is not None else "Not evaluated",
                "final_loss": self.trainer.qae_model.training_history[-1]
                if self.trainer.qae_model.training_history
                else None,
                "training_iterations": len(self.trainer.qae_model.training_history),
                "average_mse": float(np.mean(self.mse_errors)) if self.mse_errors is not None else None,
                "average_fidelity": float(np.mean(self.fidelities)) if self.fidelities is not None else None,
            }

            # Save the model
            json_path, pickle_path = self.trainer.qae_model.save_model(filepath, metadata)

            # Update the model list
            updated_choices = self.get_available_models()

            return (
                f"Model saved successfully!\n\n"
                f"• Name: {model_name}\n\n"
                f"• Files: {json_path}, {pickle_path}\n\n"
                f"• Training iterations: {len(self.trainer.qae_model.training_history)}\n\n"
                f"• Final loss: {self.trainer.qae_model.training_history[-1]:.6f}",
                gr.update(choices=updated_choices),
            )

        except Exception as e:
            return f"Failed to save model: {str(e)}", gr.update(choices=[])

    def load_model(self, selected_model):
        """Load a previously saved model"""
        if not selected_model:
            return "Please select a model to load!", None, None, gr.update(), "No images loaded", "No data available"

        try:
            models_dir = "saved_models"
            filepath = os.path.join(models_dir, selected_model)

            # Load the model using the class method to create a new trainer
            qae_model, metadata = QAE.load_from_file(filepath)

            # Create a new trainer and assign the loaded model
            self.trainer = MNIST01QAETrainer()
            self.trainer.qae_model = qae_model
            self.is_trained = True

            # Reset evaluation results
            self.originals = None
            self.predictions = None
            self.mse_errors = None
            self.fidelities = None

            # Create training plot from loaded history
            training_plot = self.create_training_plot()

            # Prepare status message
            status_msg = (
                f"Model loaded successfully!\n\n"
                f"• Description: {metadata.get('description', 'No description')}\n\n"
                f"• Training samples: {metadata.get('training_samples', 'Unknown')}\n\n"
                f"• Training iterations: {len(qae_model.training_history)}\n\n"
                f"• Final loss: {qae_model.training_history[-1]:.6f}\n\n"
                f"• Architecture: {qae_model.num_latent_qubits} latent, {qae_model.num_trash_qubits} trash qubits"
            )

            return (
                status_msg,
                training_plot,
                None,  # Clear comparison plot
                gr.update(minimum=0, maximum=9, value=0, visible=True, interactive=False),  # Reset image slider
                "No images loaded",
                "Load a model and run evaluation to view image details",
            )

        except Exception as e:
            return f"Failed to load model: {str(e)}", None, None, gr.update(), "No images loaded", "No data available"

    def get_available_models(self):
        """Get list of available saved models"""
        models_dir = "saved_models"
        if not os.path.exists(models_dir):
            return []

        # Get full paths and extract just the filenames
        full_paths = QAE.list_saved_models(models_dir)
        return [os.path.basename(path) for path in full_paths]

    def get_model_info(self, selected_model):
        """Get detailed information about a selected model"""
        if not selected_model:
            return "Select a model to view information"

        try:
            models_dir = "saved_models"
            filepath = os.path.join(models_dir, selected_model)

            # Load metadata without loading the full model
            json_filepath = f"{filepath}.json"
            with open(json_filepath, "r") as f:
                model_data = json.load(f)

            metadata = model_data.get("metadata", {})

            info = (
                f"**Model Information:**\n\n"
                f"• **Name:** {os.path.basename(selected_model)}\n\n"
                f"• **Description:** {metadata.get('description', 'No description')}\n\n"
                f"• **Architecture:** {model_data['num_latent_qubits']} latent, {model_data['num_trash_qubits']} trash qubits\n\n"
                f"• **Training Iterations:** {len(model_data['training_history'])}\n\n"
                f"• **Final Loss:** {model_data['training_history'][-1]:.6f}\n\n"
                f"• **Training Time:** {model_data['training_time']:.2f} seconds\n\n"
                f"• **Training Samples:** {metadata.get('num_sampled_trained_on', 'Unknown')}\n\n"
                f"• **Saved:** {model_data.get('saved_at', 'Unknown')}"
            )

            if metadata.get("average_mse") is not None:
                info += f"\n\n• **Average MSE:** {metadata['average_mse']:.6f}"
            if metadata.get("average_fidelity") is not None:
                info += f"\n\n• **Average Fidelity:** {metadata['average_fidelity']:.6f}"

            return info

        except Exception as e:
            return f"Error loading model info: {str(e)}"


# Initialize the demo
demo_instance = QAEDemo()


def create_interface():
    """Create the Gradio interface"""

    with gr.Blocks(title="Quantum Autoencoder MNIST Demo") as demo:
        gr.Markdown("# Quantum Autoencoder MNIST 0/1 Demo")
        gr.Markdown("Interactive visualization of quantum autoencoder results on MNIST digits (0s and 1s)")

        with gr.Tab("Training"):
            gr.Markdown("## Model Training")

            with gr.Row():
                with gr.Column(scale=1):
                    num_train = gr.Slider(
                        minimum=10, maximum=200, value=50, step=10, label="Number of Training Samples"
                    )
                    max_iterations = gr.Slider(minimum=10, maximum=200, value=50, step=10, label="Maximum Iterations")
                    train_seed = gr.Number(value=42, label="Random Seed", precision=0)
                    train_btn = gr.Button("Start Training", variant="primary")

                with gr.Column(scale=2):
                    train_status = gr.Textbox(
                        label="Training Status", placeholder="Click 'Start Training' to begin...", lines=5
                    )

            training_plot = gr.Plot(label="Training Loss History")

        with gr.Tab("Evaluation & Results"):
            gr.Markdown("## Model Evaluation")

            with gr.Row():
                with gr.Column(scale=1):
                    num_test = gr.Slider(minimum=5, maximum=50, value=10, step=5, label="Number of Test Samples")
                    eval_seed = gr.Number(value=42, label="Random Seed", precision=0)
                    eval_btn = gr.Button("Evaluate Model", variant="primary")

                with gr.Column(scale=2):
                    eval_status = gr.Textbox(
                        label="Evaluation Status",
                        placeholder="Train the model first, then click 'Evaluate Model'...",
                        lines=5,
                    )

            metrics_plot = gr.Plot(label="Metrics Distribution")

        with gr.Tab("Image Viewer"):
            gr.Markdown("## Individual Image Results")

            with gr.Row():
                with gr.Column(scale=3):
                    image_slider = gr.Slider(
                        minimum=0,
                        maximum=9,
                        value=0,
                        step=1,
                        label="Select Image Index",
                        visible=True,
                        interactive=False,
                    )
                    comparison_plot = gr.Plot()

                with gr.Column(scale=1):
                    image_counter = gr.Markdown("No images loaded")
                    image_info = gr.Markdown("Run evaluation to view image details")

        with gr.Tab("Model Management"):
            gr.Markdown("## Save and Load Models")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Save Current Model")
                    model_name = gr.Textbox(label="Model Name", placeholder="Enter a name for your model...", value="")
                    model_description = gr.Textbox(
                        label="Description", placeholder="Optional description of the model...", lines=3, value=""
                    )
                    save_btn = gr.Button("Save Model", variant="primary")
                    save_status = gr.Textbox(
                        label="Save Status", placeholder="Train a model first, then click 'Save Model'...", lines=4
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### Load Existing Model")
                    model_dropdown = gr.Dropdown(
                        choices=demo_instance.get_available_models(), label="Select Model", value=None
                    )
                    refresh_btn = gr.Button("Refresh Model List", variant="secondary")
                    load_btn = gr.Button("Load Selected Model", variant="primary")
                    load_status = gr.Textbox(
                        label="Load Status", placeholder="Select a model and click 'Load Selected Model'...", lines=4
                    )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Model Information")
                    model_info_display = gr.Markdown("Select a model to view detailed information")

        # Event handlers
        train_btn.click(
            fn=demo_instance.train_model,
            inputs=[num_train, max_iterations, train_seed],
            outputs=[train_status, training_plot],
        )

        eval_btn.click(
            fn=demo_instance.evaluate_model,
            inputs=[num_test, eval_seed],
            outputs=[eval_status, metrics_plot, comparison_plot, image_slider, image_counter, image_info],
        )

        image_slider.change(
            fn=demo_instance.update_image_display,
            inputs=[image_slider],
            outputs=[comparison_plot, image_counter, image_info],
        )

        # Model management event handlers
        save_btn.click(
            fn=demo_instance.save_model, inputs=[model_name, model_description], outputs=[save_status, model_dropdown]
        )

        load_btn.click(
            fn=demo_instance.load_model,
            inputs=[model_dropdown],
            outputs=[load_status, training_plot, comparison_plot, image_slider, image_counter, image_info],
        )

        refresh_btn.click(fn=lambda: gr.update(choices=demo_instance.get_available_models()), outputs=[model_dropdown])

        model_dropdown.change(fn=demo_instance.get_model_info, inputs=[model_dropdown], outputs=[model_info_display])

        # Footer
        gr.Markdown("---")
        gr.Markdown("Built using Gradio and Qiskit")

    return demo


if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
