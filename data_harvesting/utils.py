import torch


def get_activation_class(name):
    if name == "Tanh":
        return torch.nn.Tanh
    elif name == "ReLU":
        return torch.nn.ReLU
    elif name == "LeakyReLU":
        return torch.nn.LeakyReLU
    else:
        raise ValueError(f"Unknown activation function: {name}")