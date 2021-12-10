"""
    Defining the class to contain the feedforward neural network
"""
import logging
import torch
from pseudo_backprop.layers import FeedbackAlignmentModule
from pseudo_backprop.layers import PseudoBackpropModule
from pseudo_backprop.layers import DynPseudoBackpropModule
from pseudo_backprop.layers import VanillaBackpropModule
from pseudo_backprop import aux

logging.basicConfig(format='Network modules -- %(levelname)s: %(message)s',
                    level=logging.DEBUG)


# pylint: disable=W0223
class FullyConnectedNetwork(torch.nn.Module):
    """
        Feedforward network with relu non-linearities between the modules
    """

    def __init__(self, layers, synapse_module, net_params, mode=None):
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

        # self.synapses =[]
        # for index in range(self.num_layers - 1):
        #     # if not conv layer
        #     if not isinstance(self.layers[index], list) and not isinstance(self.layers[index + 1], list):
        #         self.synapses.append(
        #             synapse_module(self.layers[index],self.layers[index + 1])
        #             )
        #     # if next hidden layer is a conv layer
        #     elif not isinstance(self.layers[index], list) and isinstance(self.layers[index + 1], list):
        #         output_size = 
        #         self.synapses.append(
        #             synapse_module(self.layers[index], output_size)
        #             )
        #     # if current hidden layer is conv
        #     elif isinstance(self.layers[index], list) and not isinstance(self.layers[index + 1], list):
        #         input_size =
        #         self.synapses.append(
        #             synapse_module_conv2d(self.layers[index],self.layers[index + 1])
        #             )
        #      # if both are conv layers
        #     elif isinstance(self.layers[index], list) and isinstance(self.layers[index + 1], list):
        #         input_size =
        #         output_size = 
        #         self.synapses.append(
        #             synapse_module_conv2d(input_size, output_size)
        #             )
        #     else:
        #         raise ValueError(f'Parameter for layer {index} is not int or conv2d parameter array')

        self.synapses = [synapse_module(self.layers[index],
                                        self.layers[index + 1],
                                        net_params = net_params) for index in
                         range(self.num_layers - 1)]

        # look for gpu device, use gpu if available
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        # make the operations
        self.operations_list = []
        for synapse in self.synapses[:-1]:
            self.operations_list.append(synapse)
            self.operations_list.append(torch.nn.ReLU(inplace=True))
            #self.operations_list.append(torch.nn.Tanh())
        self.operations_list.append(self.synapses[-1])
        self.operations = torch.nn.Sequential(*self.operations_list)

    @classmethod
    def backprop(cls, layers, net_params):
        """
            Delegating constructor for the backprop case
        """
        logging.info("Network with vanilla backpropagation is constructed.")
        return cls(layers, VanillaBackpropModule, net_params)

    @classmethod
    def feedback_alignment(cls, layers, net_params):
        """
            Delegating constructor for the feedback alignment case
        """
        logging.info("Network with feedback alignment is constructed.")
        return cls(layers, FeedbackAlignmentModule, net_params)

    @classmethod
    def pseudo_backprop(cls, layers, net_params):
        """
            Delegating constructor for the pseudo-backprop case
        """
        logging.info("Network with pseudo-backprop is constructed.")
        return cls(layers, PseudoBackpropModule, net_params, mode='pseudo')

    @classmethod
    def gen_pseudo_backprop(cls, layers, net_params):
        """
            Delegating constructor for the generalized pseudo-backprop case
        """
        logging.info(
            "Network with generalized pseudo-backprop is constructed.")
        return cls(layers, PseudoBackpropModule, net_params, mode='gen_pseudo')

    @classmethod
    def dyn_pseudo_backprop(cls, layers, net_params):
        """
            Delegating constructor for the dynamical pseudo-backprop case
        """
        logging.info(
            "Network with dynamical pseudo-backprop is constructed.")
        return cls(layers, DynPseudoBackpropModule, net_params)

    # def conv_layer():
    #     """
    #         adds a convolutional layer
    #     """
    #     logging.info(
    #         "Convolutional 2d layer added.")

    #     return 0

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

    def redo_backward_weights(self, dataset=None, noise=None, covmat=False):
        """Recalculate the backward weights according to the model
           Do nothing if the layer has no fucntion for it.

           Add normal distributed noise if parameter noise is given.
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
                if noise:
                    logging.debug(f'Before noise: {torch.linalg.norm(input_data)}')
                    input_data = torch.normal(mean=noise[0], std=noise[1], size=list(input_data.size()))
                    logging.debug(f'After noise: {torch.linalg.norm(input_data)}')
                b_backward = aux.generalized_pseudo(
                    w_forward.detach().cpu().numpy(),
                    input_data, covmat=covmat).to(self.device)
                synapse.set_backward(b_backward)

    def get_dataspec_pinverse(self, dataset=None, covmat=False):
        """Calculate the data-specific pseudoinverse matrices.
           This function can be used to compare *any* backward matrix
           to the data-specific pseudoinverse
        """
        ds_pinv = []

        for index, synapse in enumerate(self.synapses):
            w_forward = synapse.get_forward()
            input_data = self.forward_to_hidden(dataset,
                                                index).clone().detach().cpu()
            ds_pinv.append(aux.generalized_pseudo(
                w_forward.detach().cpu().numpy(),
                input_data, covmat=covmat)
            )

        return ds_pinv

    def get_gamma_matrix(self, dataset=None):
        """Calculate Gamma matrices, i.e. the square root
           of the autocorrelation matrix of the data vectors
        """
        gamma = []

        for index, synapse in enumerate(self.synapses):
            input_data = self.forward_to_hidden(dataset,
                                                index)
            gamma.append(aux.calc_gamma_matrix(
                         input_data).detach()
            )

        return gamma

    def get_gamma2_matrix(self, dataset=None):
        """Calculate squared Gamma matrices
        """
        gamma2 = []

        for index, synapse in enumerate(self.synapses):
            input_data = self.forward_to_hidden(dataset,
                                                index)
            gamma2.append(aux.calc_gamma2_matrix_torch(
                         input_data).detach()
            )

        return gamma2

    def get_forward_weights(self):
        """Get a copy of the forward weights"""

        forward_weights = []
        for _, synapse in enumerate(self.synapses):
            weights = synapse.get_forward().detach().cpu().numpy().copy()
            forward_weights.append(weights)
        
        return forward_weights

    def get_backward_weights(self):
        """Get a copy of the backward weights"""

        forward_weights = []
        for _, synapse in enumerate(self.synapses):
            weights = synapse.get_backward().detach().cpu().numpy().copy()
            forward_weights.append(weights)

        return forward_weights
