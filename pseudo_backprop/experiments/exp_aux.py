"""Auxillary functions for the experiments."""
import argparse
import torch
from pseudo_backprop.network import FullyConnectedNetwork


def load_network(model_type, layers, net_params):
    """Load the network for testing and training

    Args:
        model_type: type of the model, string
        layers (list): number of neurons in the layers
        net_params: options for initialization of net
    """

    # make the networks
    possible_networks = ['fa', 'backprop', 'pseudo_backprop', 'gen_pseudo', 'dyn_pseudo']
    if model_type == 'fa':
        backprop_net = FullyConnectedNetwork.feedback_alignment(layers, net_params)
    elif model_type == 'backprop':
        backprop_net = FullyConnectedNetwork.backprop(layers, net_params)
    elif model_type == 'pseudo_backprop':
        backprop_net = FullyConnectedNetwork.pseudo_backprop(layers, net_params)
    elif model_type == 'gen_pseudo':
        backprop_net = FullyConnectedNetwork.gen_pseudo_backprop(layers, net_params) 
    elif model_type == 'dyn_pseudo':
        backprop_net = FullyConnectedNetwork.dyn_pseudo_backprop(layers, net_params) 
    else:
        raise ValueError(f'{model_type} is not a valid option. Implemented \
            options are in {possible_networks}')

    return backprop_net


def parse_experiment_arguments():
    """
        Parse the arguments for the test and train experiments
    """

    parser = argparse.ArgumentParser(description='Train a model on the mnist \
        dataset.')
    parser.add_argument('--params', type=str,
                        help='Path to the parameter json.')
    parser.add_argument('--dataset', type=str,
                        help='Choose from <test> or <train>.',
                        default='test')
    parser.add_argument('--per_images', type=int, default=None,
                        help='Per so many images we evaluate the model')
    parser.add_argument('--epoch', type=int,
                        help='Which epoch to validate')
    args = parser.parse_args()

    return args

def cosine_similarity_tensors(A, B):
    # Calculate the cosine similarity between two tensors
    # using the Frobenius inner product

    product = torch.trace(torch.mm(torch.t(A),B))
    norm = (torch.trace(torch.mm(torch.t(A),A)))**.5 * (torch.trace(torch.mm(torch.t(B),B)))**.5

    # returns cos(theta)
    return product / norm

# alternative measure for the distance of two tensors
def norm_distance(A,B):

    return torch.linalg.norm(A - B)**2 / torch.linalg.norm(A) / torch.linalg.norm(B)
