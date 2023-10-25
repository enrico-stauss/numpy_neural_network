import os.path

import numpy as np
from typing import Callable, Dict, Tuple, Literal, Any
from .config import Network as NetworkConfig
from .activation import ACTIVATION_FUNCTIONS_FORWARD, ACTIVATION_FUNCTIONS_BACKWARD
from .loss import LOSSES_BACKWARD
import pickle


class Layer:
    """
    A layer is defined to include the weights, biases and activation function.
    """

    def __init__(self):
        self.weights = None
        self.biases = None
        self.activation_function = None
        self.activation_forward = None
        self.activation_backward = None
        self.cache_input = None
        self.cache_z = None
        self.dz_by_dw = None
        self.dz_by_db = None

    def from_config(self, nodes_in, nodes_out, activation_function):
        self.activation_function = activation_function

        assert (
            activation_function in ACTIVATION_FUNCTIONS_BACKWARD and activation_function in ACTIVATION_FUNCTIONS_FORWARD
        )
        self.activation_forward = ACTIVATION_FUNCTIONS_FORWARD[activation_function]
        self.activation_backward = ACTIVATION_FUNCTIONS_BACKWARD[activation_function]

        self.init_weights_and_biases(nodes_in, nodes_out)

    def from_state_dict(self, state: Dict[Literal["weights", "biases", "activation"], Any]):

        self.weights = state["weights"]
        self.biases = state["biases"]
        self.activation_function = state["activation"]
        self.activation_forward = ACTIVATION_FUNCTIONS_FORWARD[self.activation_function]
        self.activation_backward = ACTIVATION_FUNCTIONS_BACKWARD[self.activation_function]

    def init_weights_and_biases(self, nodes_in, nodes_out):
        self.weights = 0.1 * np.random.randn(nodes_out, nodes_in)
        self.biases = 0.1 * np.random.randn(nodes_out)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache_input = x
        self.cache_z = self.weights.dot(x) + self.biases
        self.cache_a = self.activation_forward(self.cache_z)
        return self.cache_a

    def backward(self, output_derivative: np.ndarray):
        # Elementwise product here, because the output derivative linearly acts on the output of the activation function
        delta = output_derivative * self.activation_backward(self.cache_z, forward=self.cache_a)
        # Broadcast delta term with the cached input as each neuron in the layer has its own set of independent weights
        #    but shares the same inputs
        self.dz_by_dw = delta[:, None] * self.cache_input[None, :]
        self.dz_by_db = delta
        # Finally multiply the weights of the current layer to the returned output derivative that is propagated as the
        # dz(curr)/da(prev) term
        return delta @ self.weights

    def update(self, update_rule: Callable):
        self.weights, self.biases = update_rule(self)


def gradient_descent(layer: Layer, epsilon=1e-4):
    weights = layer.weights - epsilon * layer.dz_by_dw
    biases = layer.biases - epsilon * layer.dz_by_db
    return weights, biases


class Network:
    def __init__(self):
        self.layers = None

        self._state_forward = {}
        self._state_backward = {}

    def from_config(self, config: Dict):
        config = NetworkConfig(**config)
        model = []

        layer_previous = config.input_dimension
        for i, (layer_next, activation) in enumerate(zip(config.nodes, config.activations)):
            assert activation in ACTIVATION_FUNCTIONS_BACKWARD and activation in ACTIVATION_FUNCTIONS_FORWARD
            layer = Layer()
            layer.from_config(nodes_in=layer_previous, nodes_out=layer_next, activation_function=activation)
            model.append(layer)
            layer_previous = layer_next

        self.layers = model

    def forward(self, x: np.ndarray):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, prediction, target, cost_function: str):
        intermediate_derivative = LOSSES_BACKWARD[cost_function](input=prediction, target=target)

        for layer in reversed(self.layers):
            intermediate_derivative = layer.backward(intermediate_derivative)

    def update(self, update_rule: Callable = gradient_descent):
        for layer in self.layers:
            layer.update(update_rule)

    def save(self, filename):
        state_dict = {}
        for i, layer in enumerate(self.layers):
            state_dict[i] = {"weights": layer.weights, "biases": layer.biases, "activation": layer.activation_function}

        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        pickle.dump(state_dict, open(filename, "wb"))

    def from_checkpoint(self, filename):
        model = []
        state_dict = pickle.load(open(filename, "rb"))
        for i, parameters in state_dict.items():
            layer = Layer()
            layer.from_state_dict(parameters)
            model.append(layer)
        self.layers = model
