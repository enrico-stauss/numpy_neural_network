from typing import Literal, Tuple
import numpy as np


class DataGenerator:
    def __init__(self, function, std=0.1, index_range: Tuple = (0, 1), index_mode: Literal["discrete", "continuous"] = "continuous"):
        self.function = function
        self.std = std

        self.index_mode = index_mode
        self.index_range = index_range
        assert len(self.index_range) == 2

    def get_index(self):
        if self.index_mode == "discrete":
            raise NotImplementedError
        elif self.index_mode == "continuous":
            delta = self.index_range[1] - self.index_range[0]
            return (np.random.rand(1) * delta) - self.index_range[0]
        else:
            raise NotImplementedError

    def get_sample(self, x):
        return self.function(x)

    def get_noise(self):
        return np.random.normal(0, self.std)

