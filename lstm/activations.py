import numpy as np
from typing import Tuple


def sigmoid(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = 1 / (1 + np.exp(-z))
    da_dz = a * (1 - a)
    return a, da_dz


def tanh(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = np.tanh(z)
    da_dz = 1 - a**2
    return a, da_dz


def softmax(z: np.ndarray) -> np.ndarray:
    # Numerical stability
    exp_z = np.exp(z - np.max(z))
    a = exp_z / np.sum(exp_z)
    # No elementwise derivative here: softmax's true derivative is a full
    # Jacobian (diag(a) - a @ a.T), not an elementwise expression like the
    # other activations. When paired with cross-entropy loss, the combined
    # gradient simplifies to (y_hat - y), so we compute that directly in
    # sequence_model.py instead of differentiating softmax on its own.
    return a
