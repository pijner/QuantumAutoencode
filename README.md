# QuantumAutoencode

Exploring quantum autoencoders for image patch compression and reconstruction. This project implements quantum autoencoders using Qiskit to compress and reconstruct MNIST digit images, featuring an interactive Gradio web interface for experimentation and visualization.

## Overview

This project implements a **Quantum Autoencoder (QAE)** that learns to compress and reconstruct image data using quantum circuits. The autoencoder uses:

- **Variational Quantum Circuits**: Parameterized quantum circuits that learn optimal compression
- **Swap Test**: Quantum fidelity measurement between original and reconstructed states  
- **COBYLA Optimizer**: Classical optimization of quantum parameters
- **Amplitude Encoding**: Images encoded as quantum state amplitudes

The implementation focuses on MNIST digits (0s and 1s) and provides tools to:
- Train quantum autoencoder models with configurable parameters
- Evaluate reconstruction quality using MSE and quantum fidelity metrics
- Visualize results through an interactive web interface
- Save and load trained models for reproducible experiments

## Prerequisites

- Python 3.13
- pip (latest version recommended)

## Setup

### For Linux
Run the following commands in the `QuantumAutoencode` directory:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
```

### For Windows
Run the following commands in the `QuantumAutoencode` directory:

```bash
python -m venv venv
./venv/scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Project Structure

```
QuantumAutoencode/
├── classical/           # Classical autoencoder implementation
├── datasets/            # Dataset files and README
├── quantum/             # Quantum autoencoder implementation
│   ├── circuits/        # Quantum autoencoder circuit definitions
│   └── trainer.py       # Training logic for quantum models
├── tests/               # Unit tests
├── notebooks/           # Jupyter notebooks for development
├── saved_models/        # Saved model files (created automatically)
├── mnist_01_demo.py     # Interactive Gradio demo interface
├── requirements.txt     # Python dependencies
├── ruff.toml           # Linter configuration
└── README.md           # Project documentation
```

## Interactive Gradio Demo

This project includes an interactive web interface built with Gradio for exploring quantum autoencoder results on MNIST digits (0s and 1s).

### Launching the Demo

Run the following command from the `QuantumAutoencode` directory:

```bash
python mnist_01_demo.py
```

The interface will be available at `http://localhost:7860` in your web browser.

### Demo Features

The Gradio interface provides four main tabs:

#### 1. **Training Tab**
- Configure training parameters (number of samples, iterations, random seed)
- Train quantum autoencoder models
- View real-time training loss history with seaborn-styled plots

#### 2. **Evaluation & Results Tab**
- Evaluate trained models on test data
- View performance metrics (MSE, fidelity distributions)
- Analyze reconstruction quality statistics

#### 3. **Image Viewer Tab**
- Compare original vs reconstructed images side-by-side
- Interactive slider to browse through test samples
- Detailed metrics for each image (MSE, fidelity, norms)
- Heatmap visualizations showing reconstruction differences

#### 4. **Model Management Tab**
- **Save Models**: Save trained models with custom names and descriptions
- **Load Models**: Browse and load previously saved models
- **Model Information**: View detailed model metadata and performance metrics
- **Refresh**: Update the model list to show newly saved models

### Usage Workflow

1. **Train a Model**:
   - Go to the "Training" tab
   - Adjust parameters (50 samples, 50 iterations recommended for demo)
   - Click "Start Training" and wait for completion

2. **Evaluate Performance**:
   - Switch to "Evaluation & Results" tab
   - Set number of test samples (10-20 recommended)
   - Click "Evaluate Model" to see metrics and distributions

3. **Explore Individual Results**:
   - Go to "Image Viewer" tab
   - Use the slider to browse through reconstructed images
   - Compare original vs reconstructed vs difference heatmaps

4. **Save Your Model** (Optional):
   - Switch to "Model Management" tab
   - Enter a descriptive name and description
   - Click "Save Model" to preserve your results

5. **Load Existing Models** (Optional):
   - Select a model from the dropdown in "Model Management"
   - View model information and performance metrics
   - Click "Load Selected Model" to restore a previously trained model

### Model Persistence

Models are automatically saved with:
- Training parameters and history
- Performance metrics (MSE, fidelity)
- Architecture details
- Timestamps and custom metadata
- Both JSON (human-readable) and pickle (exact reconstruction) formats

Saved models are stored in the `saved_models/` directory and persist between sessions.

### Tips for Best Results

- Start with smaller datasets (50 training samples) for faster experimentation
- Use 50-100 iterations for reasonable convergence
- Save successful models before experimenting with different parameters
- The quantum autoencoder works best on normalized binary images (MNIST 0s and 1s)

## Unit Tests

Run the following command from the `QuantumAutoencode` directory to execute the unit tests:

```bash
python -m pytest tests
```