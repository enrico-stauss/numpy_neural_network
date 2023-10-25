import numpy as np


def binary_cross_entropy(input: np.ndarray, target: np.ndarray) -> np.ndarray:
    def log_clamped(x):
        return np.clip(np.log(x), a_min=-100)
    return -(target * log_clamped(input) + (1-input) * log_clamped(1 - input))


def binary_cross_entropy_backward(input: np.ndarray, target: np.ndarray) -> np.ndarray:
    return (input - target)/(input * (1 - input))


def mse(input: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.mean((target - input)**2)


def mse_backward(input: np.ndarray, target: np.ndarray) -> np.ndarray:
    return -2 * (target - input) / target.shape[0]


LOSSES_FORWARD = {"bce": binary_cross_entropy, "mse": mse}
LOSSES_BACKWARD = {"bce": binary_cross_entropy_backward, "mse": mse_backward}
