import os
import sys
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from source.config import load_config, Training
from source.network import Network, gradient_descent
from source.data import DataGenerator
from source.loss import LOSSES_FORWARD


def train_epoch(data_generator, network, loss, n_iters, epoch, decay_base=0.9):
    epsilon = 1e-3 * decay_base**epoch
    optimizer = partial(gradient_descent, epsilon=epsilon)
    loss_function = LOSSES_FORWARD[loss]
    index = list(range(n_iters))
    losses = []
    for i in index:
        x = data_generator.get_index()

        sample = data_generator.get_sample(x)
        sample += data_generator.get_noise()

        y = network.forward(x)

        losses.append(loss_function(y, sample))

        network.backward(prediction=y, target=sample, cost_function=loss)
        network.update(optimizer)

    return index, losses


def validate_epoch(data_generator, network, loss, n_iters):
    loss_function = LOSSES_FORWARD[loss]
    index = list(range(n_iters))
    losses = []
    for i in index:
        x = data_generator.get_index()
        sample = data_generator.get_sample(x)
        y = network.forward(x)
        losses.append(loss_function(y, sample))
    losses = np.mean(np.array(losses))
    return losses


def main():
    # Some experiment settings can be adjusted here:
    config_file = "config/config_1.yaml"

    def target_fn(x):
        return 2*x**2 + 3 * x + 4

    data_generator = DataGenerator(function=target_fn, std=0.01, index_range=(-np.pi, np.pi), index_mode="continuous")

    # Do not modify below, all other configuration happens in the config file
    config = load_config(config_file)
    training_config = Training(**config["TRAINING"])
    network = Network()

    if training_config.checkpoint_load is not None:
        network.from_checkpoint(training_config.checkpoint_load)
    else:
        network.from_config(config["NETWORK"])

    training_step = []
    validation_step = []

    training_loss = []
    validation_loss = []

    kwargs = {
        "data_generator": data_generator,
        "network": network,
        "loss": training_config.loss_function,
        "n_iters": training_config.steps_per_epoch,
    }

    for i in tqdm.trange(training_config.train_epochs):
        if i % training_config.validation_frequency == 0:
            validation_loss.append(validate_epoch(**kwargs))
            validation_step.append(i * training_config.steps_per_epoch)

        index, loss = train_epoch(**kwargs, epoch=i, decay_base=training_config.epsilon_decay_base)
        training_loss.extend(loss)
        training_step.extend([j + i * training_config.steps_per_epoch for j in index])

    else:
        validation_loss.append(validate_epoch(**kwargs))
        validation_step.append(training_config.train_epochs * training_config.steps_per_epoch)

    if training_config.checkpoint_write:
        network.save(training_config.checkpoint_write)

    plt.plot(training_step, training_loss, label="train loss")
    plt.plot(validation_step, validation_loss, label="validation loss")
    plt.yscale("log")
    plt.show()


if __name__ == "__main__":
    main()
