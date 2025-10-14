# QuantumAutoencode

Exploring quantum autoencoders for image patch compression and reconstruction. This project compares quantum and classical autoencoders using Qiskit and TensorFlow to evaluate compression fidelity, resource efficiency, and potential for quantum-enhanced anomaly detection.

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
├── classical/        # Classical autoencoder implementation
├── datasets/         # Dataset files and README
├── quantum/          # Quantum autoencoder implementation
├── tests/            # Unit tests
├── requirements.txt  # Python dependencies
├── ruff.toml         # Linter configuration
└── README.md         # Project documentation
```

## Unit Tests

Run the following command from the `QuantumAutoencode` directory to execute the unit tests:

```bash
python -m pytest tests
```