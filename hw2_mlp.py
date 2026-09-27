"""Starter implementation for CS 577 Homework 2.

Replace each NotImplementedError with your own NumPy implementation. Do not
change function names, arguments, dictionary keys, or return-value order: the
public and hidden tests use this interface.

The manual MLP functions must not use PyTorch, JAX, TensorFlow, autograd
packages, symbolic differentiation, or finite differences.
"""

from typing import Dict, List, Tuple

import numpy as np


ArrayDict = Dict[str, np.ndarray]
PARAMETER_KEYS = ("W1", "b1", "W2", "b2")


def sigmoid_stable(z: np.ndarray) -> np.ndarray:
    """Elementwise sigmoid computed without overflow."""
    return np.exp(-np.logaddexp(0.0, -z)) 


def relu(z: np.ndarray) -> np.ndarray:
    """Elementwise ReLU."""
    return np.maximum(0,z)


def binary_cross_entropy_with_logits(
    logits: np.ndarray, y: np.ndarray
) -> float:
    """Return mean binary cross-entropy from logits.

    Both arrays must have identical shape (B, 1). Use the numerically stable
    BCE-with-logits expression stated in the assignment.
    """
    if logits.shape != y.shape:
        raise ValueError("logits and y must have the same shape")

    if logits.ndim != 2 or logits.shape[1] != 1:
        raise ValueError("logits and y must have shape (B, 1)")

    loss=np.maximum(0,logits)-y*logits+np.log1p(np.exp(-np.abs(logits)))

    return float(np.mean(loss))


def initialize_parameters(
    input_dim: int,
    hidden_dim: int,
    seed: int = 577,
    weight_scale: float = 0.8,
) -> ArrayDict:
    """Return deterministic float64 parameters for a D-H-1 MLP.

    This helper is provided so every student uses the same initialization
    convention in the required experiment.
    """
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0.0, weight_scale, size=(input_dim, hidden_dim)),
        "b1": np.zeros(hidden_dim, dtype=np.float64),
        "W2": rng.normal(0.0, weight_scale, size=(hidden_dim, 1)),
        "b2": np.zeros(1, dtype=np.float64),
    }


def mlp_forward(
    X: np.ndarray, params: ArrayDict
) -> Tuple[np.ndarray, ArrayDict]:
    """Run affine -> ReLU -> affine and return (logits, cache)."""
    Z1=X@params["W1"]+params["b1"]
    A1=relu(Z1)
    S=A1@params["W2"]+params["b2"]
    cache = {
        "X": X,
        "Z1": Z1,
        "A1": A1,
        }

    return S, cache


def mlp_loss_and_gradients(
    X: np.ndarray, y: np.ndarray, params: ArrayDict
) -> Tuple[float, ArrayDict]:
    """Return mean BCE-with-logits loss and manual parameter gradients."""
    S, cache = mlp_forward(X, params)
    loss = binary_cross_entropy_with_logits(S, y)
    B = X.shape[0]
    dS = (sigmoid_stable(S) - y) / B
    A1 = cache["A1"]
    dW2 = A1.T @ dS
    db2 = np.sum(dS, axis=0)
    dA1 = dS @ params["W2"].T
    dZ1=dA1*(cache["Z1"]>0)
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)
    grads = {
        "W1":dW1 ,
        "b1": db1,
        "W2": dW2,
        "b2": db2
    }

    return loss, grads


def finite_difference_gradients(
        X: np.ndarray,
        y: np.ndarray,
        params: ArrayDict,
        epsilon: float = 1e-5,
    ) -> ArrayDict:
        """Return centered finite-difference gradients for all scalar parameters."""
        numerical_grads = {
            key: np.zeros_like(value)
            for key, value in params.items()
        }

        # Work on a copy so the original params dictionary stays unchanged
        params_work = {
            key: value.copy()
            for key, value in params.items()
        }

        for key in params_work:
            for idx in np.ndindex(params_work[key].shape):

                original_value = params_work[key][idx]

                # 1. Set parameter to original + epsilon
                params_work[key][idx] = original_value + epsilon
                J_plus=mlp_loss_and_gradients(X,y, params_work)[0]


                # 2. Set parameter to original - epsilon
                params_work[key][idx] = original_value - epsilon
                J_minus=mlp_loss_and_gradients(X,y, params_work)[0]


                # 3. Centered finite difference
                numerical_grads[key][idx] = (
                    J_plus - J_minus
                ) / (2 * epsilon)

                # 4. Restore original parameter
                params_work[key][idx] = original_value

        return numerical_grads
    
def relative_error(a, b):
    numerator = np.linalg.norm(a - b)
    denominator = np.linalg.norm(a) + np.linalg.norm(b) + 1e-12
    return float(numerator / denominator)


def predict_proba(X: np.ndarray, params: ArrayDict) -> np.ndarray:
    """Return P(y=1|x), shape (B, 1)."""
    logits, _ = mlp_forward(X, params)
    return sigmoid_stable(logits)


def predict_labels(
    X: np.ndarray, params: ArrayDict, threshold: float = 0.5
) -> np.ndarray:
    """Return integer predictions with shape (B, 1)."""
    return (predict_proba(X, params) >= threshold).astype(np.int64)


def binary_accuracy(y_pred: np.ndarray, y: np.ndarray) -> float:
    """Return fraction of equal binary labels after validating shape."""
    if y_pred.shape != y.shape:
        raise ValueError(f"shape mismatch: predictions {y_pred.shape}, targets {y.shape}")
    return float(np.mean(y_pred == y))


def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    hidden_dim: int = 8,
    steps: int = 2000,
    lr: float = 0.2,
    seed: int = 577,
) -> Tuple[ArrayDict, List[float]]:
    """Train the manual MLP with full-batch gradient descent."""
    raise NotImplementedError


def make_xor_data(
    n_per_quadrant: int = 60,
    noise: float = 0.28,
    seed: int = 577,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a deterministic noisy XOR-style binary dataset.

    This data helper is provided and is not a TODO. The four cluster centers
    alternate labels so no single straight decision boundary can separate all
    classes.
    """
    if n_per_quadrant <= 0:
        raise ValueError("n_per_quadrant must be positive")
    if noise < 0:
        raise ValueError("noise must be nonnegative")
    rng = np.random.default_rng(seed)
    centers = np.array(
        [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    labels = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64)
    X_parts = [
        center + rng.normal(0.0, noise, size=(n_per_quadrant, 2))
        for center in centers
    ]
    y_parts = [np.full((n_per_quadrant, 1), label) for label in labels]
    X = np.vstack(X_parts)
    y = np.vstack(y_parts)
    order = rng.permutation(X.shape[0])
    return X[order], y[order]
