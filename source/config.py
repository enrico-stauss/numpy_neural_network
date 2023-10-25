import pydantic
from typing import Literal, List, Optional
import yaml


def load_config(filename):
    return yaml.load(open(filename, "r"), Loader=yaml.FullLoader)


class Network(pydantic.BaseModel):
    input_dimension: int
    activations: List[Literal["ReLu", "Sigmoid", "Identity", "Sin"]]
    nodes: List[int]

    @pydantic.model_validator(mode="after")
    def check_length(self):
        assert len(self.activations) == len(self.nodes)


class Training(pydantic.BaseModel):
    train_epochs: int
    steps_per_epoch: int
    validation_frequency: int = 1
    loss_function: Literal["mse", "bce"] = "mse"
    epsilon_decay_base: float = 0.9
    checkpoint_load: Optional[str] = None
    checkpoint_write: Optional[str] = None

    @pydantic.field_validator("checkpoint_load", "checkpoint_write")
    def validate_checkpoint_load(cls, value):
        if isinstance(value, str) and value == "None":
            value = None
        return value
