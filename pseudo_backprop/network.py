"""
    Defining the class to contain the feedforward neural network
"""
import logging
import torch
from pseudo_backprop.layers import FeedbackAlginementModule
from pseudo_backprop.layers import PseudoBackpropModule
from pseudo_backprop.layers import VanillaLinear
from pseudo_backprop import aux

logging.basicConfig(format='Network modules -- %(levelname)s: %(message)s',
                    level=logging.DEBUG)


# pylint: disable=W0223
class FullyConnectedNetwork(torch.nn.Module):
    """
        Feedforward network with relu non-linearities between the modules
    """

    def __init__(self, layers, synapse_module, mode=None):
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
        if mode is not None:
            self.mode = mode
        self.synapses = [synapse_module(self.layers[index],
                                        self.layers[index + 1]) for index in
                         range(self.num_layers - 1)]

        # look for gpu device, use gpu if available
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        # make the operations
        self.operations_list = []
        for synapse in self.synapses:
            self.operations_list.append(synapse)
            self.operations_list.append(torch.nn.ReLU(inplace=True))
        self.operations = torch.nn.Sequential(*self.operations_list)

    @classmethod
    def backprop(cls, layers):
        """
            Delegating constructor for the backprop case
        """
        logging.info("Network with vanilla backpropagation is constructed.")
        return cls(layers, VanillaLinear)

    @classmethod
    def feedback_alignement(cls, layers):
        """
            Delegating constructor for the feedback alignement case
        """
        logging.info("Network with feedback alignement is constructed.")
        return cls(layers, FeedbackAlginementModule)

    @classmethod
    def pseudo_backprop(cls, layers):
        """
            Delegating constructor for the pseudo-backprop case
        """
        logging.info("Network with pseudo-backprop is constructed.")
        return cls(layers, PseudoBackpropModule, mode='pseudo')

    @classmethod
    def gen_pseudo_backprop(cls, layers):
        """
            Delegating constructor for the generalized pseudo-backprop case
        """
        logging.info(
            "Network with generalized pseudo-backprop is constructed.")
        return cls(layers, PseudoBackpropModule, mode='gen_pseudo')

    @classmethod
    def dyn_pseudo_backprop(cls, layers):
        """
            Delegating constructor for the dynamical pseudo-backprop case
        """
        logging.info(
            "Network with dynamical pseudo-backprop is constructed.")
        return cls(layers, PseudoBackpropModule, mode='gen_pseudo')

    def forward(self, inputs):
        """
            Entire calculation of the model
        """

        return self.operations(inputs)

    def forward_to_hidden(self, inputs, layer):
        """
        Make a forward pass on the inputs to the layer-th
        evaluation

        Args:
            inputs (tensor): tensor of inputs
            layer (int): layer number, if layer==0 then the
                         input is returned

        Returns:
            tensor: Activities in the layer-th layer
        """

        if layer == 0:
            return inputs

        # each layer is a combination of a matrix vector multiplication
        # and a non-linearity
        for index in range(2 * layer):
            inputs = self.operations_list[index](inputs)

        return inputs

    def redo_backward_weights(self, dataset=None):
        """Recalculate the backward weights according to the model
           Do nothing if the layer has no fucntion for it.
        """

        if self.mode == 'pseudo':
            for synapse in self.synapses:
                synapse.redo_backward()
        elif self.mode == 'gen_pseudo':
            logging.info('gen_pseudo redo was called')
            for index, synapse in enumerate(self.synapses):
                logging.info(f'(Re-)calculating backward weights in layer: {index}')
                w_forward = synapse.get_forward()
                input_data = self.forward_to_hidden(dataset,
                                                    index)
                b_backward = aux.generalized_pseudo(
                    w_forward.detach().cpu().numpy(),
                    input_data).to(self.device)
                synapse.set_backward(b_backward)

    def get_forward_weights(self):
        """Get a copy of the forward weights"""

        forward_weights = []
        for _, synapse in enumerate(self.synapses):
            weights = synapse.get_forward().detach().numpy().copy()
            forward_weights.append(weights)

        return forward_weights

    def get_backward_weights(self):
        """Get a copy of the backward weights"""

        forward_weights = []
        for _, synapse in enumerate(self.synapses):
            weights = synapse.get_backward().detach().numpy().copy()
            forward_weights.append(weights)

        return forward_weights
