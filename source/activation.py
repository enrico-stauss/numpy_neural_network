from typing import Optional
import numpy as np


def relu(z: np.ndarray) -> np.ndarray:
    #zeros = np.zeros_like(z)
    return np.maximum(0, z)


def relu_backward(z: np.ndarray, forward: Optional[np.ndarray]) -> np.ndarray:
    return np.where(z < 0, 0, 1)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1/(1+np.exp(-z))


def sigmoid_backward(z: np.ndarray, forward: Optional[np.ndarray]) -> np.ndarray:
    return forward * (1 - forward)


def identity(z: np.ndarray) -> np.ndarray:
    return z


def identity_backward(z: np.ndarray, forward: Optional[np.ndarray]) -> np.ndarray:
    return np.ones_like(z)


def sin(z: np.ndarray) -> np.ndarray:
    return np.sin(z)


def sin_backward(z: np.ndarray, forward: Optional[np.ndarray]) -> np.ndarray:
    return np.cos(z)


ACTIVATION_FUNCTIONS_FORWARD = {
    "ReLu": relu,
    "Sigmoid": sigmoid,
    "Identity": identity,
    "Sin": sin,
}


ACTIVATION_FUNCTIONS_BACKWARD = {
    "ReLu": relu_backward,
    "Sigmoid": sigmoid_backward,
    "Identity": identity_backward,
    "Sin": sin_backward,
}
