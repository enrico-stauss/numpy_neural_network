import numpy as np
from typing import Callable


def layer_forward(input: np.ndarray, W: np.ndarray, b: np.ndarray, activation: Callable) -> np.ndarray:
    assert W.shape[-1] == input.shape[0]
    return activation(W.dot(input) + b)
