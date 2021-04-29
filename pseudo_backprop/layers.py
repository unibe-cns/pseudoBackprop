"""
    Modules that define the backward path for the feedback alignment
    and the pseudo-prop linearities

    Structure:
    We define first the passes (forward/backward)
    and then the modules for each modified algorithm

    The relevant variables for the passes are:
    grad_input: this is the gradient of the cost w.r.t. the weighted input
                of the layer, i.e. the error delta_{l-1} in the standard notation
    grad_output: gradient of cost w.r.t. the output of each layer, delta_l
    grad_weight: gradient of cost w.r.t. weight matrix
    weight: forward weight matrix W
    back_weight: backward weight matrix B
                 for backprop, B = W^T
                 for FA, B = random fixed matrix
                 for dyn pseudo backprop, we update B according to a learning rule

    Note that this module omits all activation functions and derivatives.
    This is handles separately by torch, see network.py
"""
import logging
import math
import torch
import numpy as np
from torch import nn

logging.basicConfig(format='Layer modules -- %(levelname)s: %(message)s',
                    level=logging.DEBUG)
SCALING_FACTOR = 4

# pylint: disable=W0223,W0212
class VanillaLinear(torch.nn.Linear):
    """Vanilla Linear
       inherit from the torch.nn.Linear to make the init possible
    """

    def reset_parameters(self) -> None:
        """reset and/or init the parameters
           largely taken from pytorch
        """
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
            self.weight)
        bound = math.sqrt(SCALING_FACTOR / fan_in)
        torch.nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            torch.nn.init.uniform_(self.bias, -bound, bound)


# The feedback alignment components
# The following two functions inherit from torch functionalities to realize
# the feedback alignement.

# pylint: disable=W0223
class FeedbackAlignmentLinearity(torch.autograd.Function):
    """
        The feedack alignment function
        This defines the forward and the backwards directions
    """

    # pylint: disable=W0221
    @staticmethod
    def forward(ctx, input_torch, weight, back_weight, bias=None):
        """
         the forward calculation

         Params:
            ctx: context object to save variables for the backward pass
            input_torch: the input tensor
            weight: the forward weight matrix
            back_weight: the backward weight matrix which is fix during learning
            bias: tensor of the bias variables if applicable
        """

        ctx.save_for_backward(input_torch, weight, back_weight, bias)
        output = input_torch.mm(weight.t())
        if bias is not None:
            output += torch.unsqueeze(bias, 0).expand_as(output)

        return output

    # pylint: disable=W0221
    @staticmethod
    def backward(ctx, grad_output):
        """
            calculate the necessary gradients

            Params:
            ctx: context object to save variables for the backward pass
            grad_output: current gradient at the output of the forward pass
        """

        # get variables from the forward pass
        input_torch, _, back_weight, bias = ctx.saved_variables
        grad_back_weight = None

        # calculate the gradients that are backpropagated
        grad_input = grad_output.mm(back_weight)
        # calculate the gradients on the weights
        grad_weight = grad_output.t().mm(input_torch)
        if (bias is not None) and (ctx.needs_input_grad[3]):
            # gradient at the bias if required
            grad_bias = grad_output.sum(0).squeeze(0)
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_back_weight, grad_bias


# pylint: disable=R0903
class FeedbackAlignmentModule(nn.Module):
    """
        Define a module of synapses for the feedback alignement synapses
    """

    def __init__(self, input_size, output_size, bias=True):
        """
            feedback alignement module with initilaization

            Params:
            input_size: input size of the module
            output_size: output size
                         The module represents a linear map of the size
                         input_size X output_size
        """

        # call parent for proper init
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        if bias:
            logging.info('Bias is activated')
        else:
            logging.info('Bias is deactivated.')

        # create the parameters
        self.weight = nn.Parameter(torch.Tensor(self.output_size,
                                                self.input_size),
                                   requires_grad=True)
        # create the biases if applicable
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_size),
                                     requires_grad=True)
        else:
            self.register_buffer('bias', None)

        # create a variable for the random feedback weights
        self.weight_back = nn.Parameter(
            torch.FloatTensor(self.output_size,
                              self.input_size),
            requires_grad=False)

        # Initialize the weights
        k_init = np.sqrt(SCALING_FACTOR/self.input_size)
        torch.nn.init.uniform_(self.weight, a=-1 * k_init,
                               b=k_init)
        torch.nn.init.uniform_(self.weight_back, a=-1*k_init,
                               b=k_init)
        if bias:
            torch.nn.init.uniform_(self.bias, a=-SCALING_FACTOR*k_init,
                                   b=SCALING_FACTOR * k_init)

    def forward(self, input_tensor):
        """
            Method to calculate the forward processing through the synapses
        """
        # the forward calcualtion of the module
        return FeedbackAlignmentLinearity.apply(input_tensor,
                                                self.weight,
                                                self.weight_back,
                                                self.bias)


# The pseudo backpropagation components
# The following two functions inherit from torch functionalities to realize
# the pseudo backprop.


# pylint: disable=W0223
class PseudoBackpropLinearity(torch.autograd.Function):
    """
        The pseudobackprop function
        This defines the forward and the backwards directions
    """

    # pylint: disable=W0221
    @staticmethod
    def forward(ctx, input_torch, weight, back_weight, bias=None):
        """
         the forward calculation

         Params:
            ctx: context object to save variables for the backward pass
            input_torch: the input tensor
            weight: the forward weight matrix
            back_weight: the backward weight matrix
            bias: tensor of the bias variables if applicable
        """

        ctx.save_for_backward(input_torch, weight, back_weight, bias)
        output = input_torch.mm(weight.t())
        if bias is not None:
            output += torch.unsqueeze(bias, 0).expand_as(output)

        return output

    # pylint: disable=W0221
    @staticmethod
    def backward(ctx, grad_output):
        """
            calculate the necessary gradients

            Params:
            ctx: context object to save variables for the backward pass
            grad_output: current gradient at the output of the forward pass
        """

        # get variables from the forward pass
        input_torch, weight, back_weight, bias = ctx.saved_variables
        grad_back_weight = None

        # calculate the gradients that are backpropagated
        # using the pinv that has been calculated before:
        # pseudo_backprop: pinv == linalg.pinv
        # gen_pseudo: pinv == generalized pseudo, see aux.py
        pseudo_inverse = back_weight
        grad_input = grad_output.mm(pseudo_inverse)
        # calculate the gradients on the weights
        grad_weight = grad_output.t().mm(input_torch)
        if (bias is not None) and (ctx.needs_input_grad[3]):
            # gradient at the bias if required
            grad_bias = grad_output.sum(0).squeeze(0)
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_back_weight, grad_bias


# pylint: disable=R0903
class PseudoBackpropModule(nn.Module):
    """
        Define a module of synapses for the pseudo backprop synapses
    """

    def __init__(self, input_size, output_size, bias=True):
        """
            pseudobackprop module with initilaization

            Params:
            input_size: input size of the module
            output_size: output size
                         The module represents a linear map of the size
                         input_size X output_size
        """

        # call parent for proper init
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.counter = 0
        if bias:
            logging.info('Bias is activated')
        else:
            logging.info('Bias is deactivated.')

        # create the parameters
        self.weight = nn.Parameter(torch.Tensor(self.output_size,
                                                self.input_size),
                                   requires_grad=True)

        # create the biases if applicable
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_size),
                                     requires_grad=True)
        else:
            self.register_buffer('bias', None)

        # Initialize the weights
        k_init = np.sqrt(SCALING_FACTOR/self.input_size)
        torch.nn.init.uniform_(self.weight, a=-1*k_init,
                               b=k_init)
        # KM: this is not the correct backweight matrix for gen_pseudo!
        self.pinv = nn.Parameter(torch.linalg.pinv(self.weight),
                                 requires_grad=False)

        if bias:
            torch.nn.init.uniform_(self.bias, a=-1*k_init,
                                   b=k_init)

    def forward(self, input_tensor):
        """
            Method to calculate the forward processing through the synapses
        """

        return PseudoBackpropLinearity.apply(input_tensor,
                                                self.weight,
                                                self.pinv.t(),
                                                self.bias)

    def redo_backward(self):
        """Recalculate the matrix that is used for the backwards direction
        """
        logging.debug('Redo backward called')
        self.pinv = nn.Parameter(torch.linalg.pinv(self.weight.detach()),
                                 requires_grad=False)

    def set_backward(self, backward):
        """Set the backward synapses from the outside

        Args:
            backward (torch.tensor): Description
        """

        self.pinv = nn.Parameter(
            backward.float(), requires_grad=False)

    def get_forward(self):
        """Get a detached clone of the forward weights

        Returns:
            torch.tensor: The forward weights
        """

        return self.weight.clone().detach()

    def get_backward(self):
        """Get a detached clone of the forward weights

        Returns:
            torch.tensor: The backward weights
        """

        return self.pinv.clone().detach()


# The _dynamical_ pseudo backpropagation components
# The following two functions inherit from torch functionalities to realize
# dynamical pseudo backprop.


# pylint: disable=W0223
class DynPseudoBackpropLinearity(torch.autograd.Function):
    """
        The dynamical pseudobackprop function
        This defines the forward and the backwards directions
    """

    # pylint: disable=W0221
    @staticmethod
    def forward(ctx, input_torch, weight, back_weight, bias=None):
        """
         the forward calculation
         this is a standard forward pass

         Params:
            ctx: context object to save variables for the backward pass
            input_torch: the input tensor
            weight: the forward weight matrix
            back_weight: the backward weight matrix
            bias: tensor of the bias variables if applicable
        """

        ctx.save_for_backward(input_torch, weight, back_weight, bias)
        output = input_torch.mm(weight.t())
        if bias is not None:
            output += torch.unsqueeze(bias, 0).expand_as(output)

        return output

    # pylint: disable=W0221
    @staticmethod
    def backward(ctx, grad_output):
        """
            calculate the necessary gradients
            this is where we implement _two independent_ weight updates

            Params:
            ctx: context object to save variables for the backward pass
            grad_output: current gradient at the output of the forward pass
        """

        # get variables from the forward pass
        input_torch, weight, back_weight, bias = ctx.saved_variables
        grad_back_weight = None

        # calculate the gradients that are backpropagated
        grad_input = grad_output.mm(back_weight)
        # calculate the gradients on the weights
        grad_weight = grad_output.t().mm(input_torch)
        if (bias is not None) and (ctx.needs_input_grad[3]):
            # gradient at the bias if required
            grad_bias = grad_output.sum(0).squeeze(0)
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_back_weight, grad_bias


# pylint: disable=R0903
class DynPseudoBackpropModule(nn.Module):
    """
        Define a module of synapses for dynamical pseudo backprop synapses
    """

    def __init__(self, input_size, output_size, bias=True):
        """
            dynamical pseudobackprop module with initilaization

            Params:
            input_size: input size of the module
            output_size: output size
                         The module represents a linear map of the size
                         input_size X output_size
        """

        # call parent for proper init
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.counter = 0
        if bias:
            logging.info('Bias is activated')
        else:
            logging.info('Bias is deactivated.')

        # create the parameters
        self.weight = nn.Parameter(torch.Tensor(self.output_size,
                                                self.input_size),
                                   requires_grad=True)

        # create a variable for the feedback weights
        # these are going to be dynamical, so we require grad
        self.weight_back = nn.Parameter(
            torch.FloatTensor(self.output_size,
                              self.input_size),
            requires_grad=True)

        # create the biases if applicable
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_size),
                                     requires_grad=True)
        else:
            self.register_buffer('bias', None)

        # Initialize the weights
        k_init = np.sqrt(SCALING_FACTOR/self.input_size)
        torch.nn.init.uniform_(self.weight, a=-1*k_init,
                               b=k_init)
        torch.nn.init.uniform_(self.weight_back, a=-1*k_init,
                               b=k_init)

        if bias:
            torch.nn.init.uniform_(self.bias, a=-1*k_init,
                                   b=k_init)

    def forward(self, input_tensor):
        """
            Method to calculate the forward processing through the synapses
        """

        return DynPseudoBackpropLinearity.apply(input_tensor,
                                                self.weight,
                                                self.weight_back,
                                                self.bias)

    def get_forward(self):
        """Get a detached clone of the forward weights

        Returns:
            torch.tensor: The forward weights
        """

        return self.weight.clone().detach()

    def get_backward(self):
        """Get a detached clone of the backward weights

        Returns:
            torch.tensor: The backward weights
        """

        return self.weight_back.clone().detach()
