import numpy as np


def binary_cross_entropy(input: np.ndarray, target: np.ndarray) -> np.ndarray:
    def log_clamped(x):
        return np.clip(np.log(x), a_min=-100)
    return -(target * log_clamped(input) + (1-input) * log_clamped(1 - input))
