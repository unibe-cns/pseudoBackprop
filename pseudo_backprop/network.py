"""
    Defining the class to contain the feedforward neural network
"""
import logging
import torch
from pseudo_backprop.layers import FeedbackAlginementModule

logging.basicConfig(format='Network modules -- %(levelname)s: %(message)s',
                    level=logging.DEBUG)


# pylint: disable=W0223
class FullyConnectedNetwork(torch.nn.Module):
    """
        Feedforward network with relu non-linearities between the modules
    """

    def __init__(self, layers, synapse_module):
        """
            Initialize the network

            Params:
                layers: the structur of the model
                        [input, hidden1, hidden2, ... , hiddenN, output]
                synapse_module: connection between the layers
                                it makes a difference in the backwards pass
        """

        # call parent for proper init
        super().__init__()
        self.num_layers = len(layers)
        self.layers = layers

        # create the synapse
        self.synapses = [synapse_module(self.layers[index],
                                        self.layers[index + 1]) for index in
                         range(self.num_layers - 1)]

        # make the operations
        self.operations = []
        for synapse in self.synapses:
            self.operations.append(synapse)
            self.operations.append(torch.nn.LeakyReLU(negative_slope=0.05,
                                                      inplace=True))
        self.operations = torch.nn.Sequential(*self.operations)

    @classmethod
    def backprop(cls, layers):
        """
            Delegating constructor for the backprop case
        """
        logging.info("Network with backpropagation is constructed.")
        return cls(layers, torch.nn.Linear)

    @classmethod
    def feedback_alignement(cls, layers):
        """
            Delegating constructor for the backprop case
        """
        logging.info("Network with feedback alignement is constructed.")
        return cls(layers, FeedbackAlginementModule)

    def forward(self, inputs):
        """
            Entire calculation of the model
        """

        return self.operations(inputs)
