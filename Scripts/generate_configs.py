import itertools


def generate_configs():
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [16, 32, 64]
    epochs = [10, 20, 30]
    configs = list (itertools.product (learning_rates, batch_sizes, epochs))
    return configs
