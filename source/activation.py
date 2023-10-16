import numpy as np


def relu(z: np.ndarray) -> np.ndarray:
    zeros = np.zeros_like(z)
    return np.maximum(zeros, z)


def relu_backward(da: np.ndarray, z: np.ndarray) -> np.ndarray:
    dz = da.copy()
    dz[z <= 0] = 0
    return dz


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1/(1+np.exp(-z))


def sigmoid_backward(da: np.ndarray, z: np.ndarray) -> np.ndarray:
    sig_z = np.sigmoid(z)
    return da * sig_z * (1 - sig_z)