"""Auxillary functions for the experiments."""
from pseudo_backprop.network import FullyConnectedNetwork


def load_network(model_type, layers):
    """Load the network for testing and training

    Args:
        model_type: type of the model, string
        layers (list): number of neurons in the layers
    """

    # make the networks
    possible_networks = ['fa', 'backprop', 'pseudo_backprop']
    if model_type == 'fa':
        backprop_net = FullyConnectedNetwork.feedback_alignement(layers)
    elif model_type == 'backprop':
        backprop_net = FullyConnectedNetwork.backprop(layers)
    elif model_type == 'pseudo_backprop':
        backprop_net = FullyConnectedNetwork.pseudo_backprop(layers)
    else:
        raise ValueError(f'{model_type} is not a valid option. Implemented \
            options are in {possible_networks}')

    return backprop_net
