"""
    Defining the class to contain the feedforward neural network
"""
import torch


class FullyConnectedNetwork(torch.nn.Module):
    """
        Feedforward network with relu non-linearities between the modules
    """

    def __init__(self, layers, synapseModule):
        """
            Initialize the network

            Params:
                layers: the structur of the model
                        [input, hidden1, hidden2, ... , hiddenN, output]
                synapseModule: connection between the layers
                               it makes a difference in the backwards pass
        """

        # call parent for proper init
        super().__init__()
        self.num_layers = len(layers)
        self.layers = layers

        # create the synapse
        self.synapses = [synapseModule(self.layers[index], self.layers[index +
                         1]) for index in range(self.num_layers - 1)]

        # make the operations
        self.operations = []
        for synapse in self.synapses:
            self.operations.append(synapse)
            self.operations.append(torch.nn.ReLU(inplace=True))
        self.operations = torch.nn.Sequential(*self.operations)

    @classmethod
    def backprop(cls, layers):
        """
            Delegating constructor for the backprop case
        """
        return cls(layers, torch.nn.Linear)

    def forward(self, inputs):
        """
            Entire calculation of the model
        """

        return self.operations(inputs)
