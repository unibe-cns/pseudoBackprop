"""
    Modules that define the backward path for the feedback alignment
    and the pseudo-prop lienarities
"""
import logging
import torch
from torch import nn

logging.basicConfig(format='Layer modules -- %(levelname)s: %(message)s',
                    level=logging.DEBUG)


# The feedback algnement components
# The following two functions inherit from torch functionalities to relaize
# the feedback alignement.


# pylint: disable=W0223
class FeedbackAlignmentLinearity(torch.autograd.Function):
    """
        The feedack alignment function
        This defines the forward and the backwords directions
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
            back_weight: the backward weight matrix which is fix during learing
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
class FeedbackAlginementModule(nn.Module):
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
        self.weight_back = torch.autograd.Variable(
            torch.FloatTensor(self.output_size,
                              self.input_size),
            requires_grad=False)

        # Initialize the weights
        torch.nn.init.kaiming_normal_(self.weight, mode='fan_in',
                                      nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.weight_back, mode='fan_in',
                                      nonlinearity='relu')
        if bias:
            torch.nn.init.normal_(self.bias)

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
# The following two functions inherit from torch functionalities to relaize
# the pseudo backprop.


# pylint: disable=W0223
class PseudoBackpropLinearity(torch.autograd.Function):
    """
        The feedack alignment function
        This defines the forward and the backwords directions
    """

    # pylint: disable=W0221
    @staticmethod
    def forward(ctx, input_torch, weight, bias=None):
        """
         the forward calculation

         Params:
            ctx: context object to save variables for the backward pass
            input_torch: the input tensor
            weight: the forward weight matrix
            bias: tensor of the bias variables if applicable
        """

        ctx.save_for_backward(input_torch, weight, bias)
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
        input_torch, weight, bias = ctx.saved_variables

        # calculate the gradients that are backpropagated
        # use the pseudoinverse for the backward path
        # this is a time-consuming operation
        pseudo_inverse = torch.pinverse(weight, rcond=1e-10)
        grad_input = grad_output.mm(pseudo_inverse.t())
        # calculate the gradients on the weights
        grad_weight = grad_output.t().mm(input_torch)
        if (bias is not None) and (ctx.needs_input_grad[2]):
            # gradient at the bias if required
            grad_bias = grad_output.sum(0).squeeze(0)
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias


# pylint: disable=R0903
class PseudoBackpropModule(nn.Module):
    """
        Define a module of synapses for the pseudo backprop synapses
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
        torch.nn.init.kaiming_normal_(self.weight, mode='fan_in',
                                      nonlinearity='relu')
        self.pinv = nn.Parameter(torch.pinverse(self.weight, rcond=1e-10),
                                 requires_grad=False)
        if bias:
            torch.nn.init.normal_(self.bias)

    def forward(self, input_tensor):
        """
            Method to calculate the forward processing through the synapses
        """

        return FeedbackAlignmentLinearity.apply(input_tensor,
                                                self.weight,
                                                self.pinv.t(),
                                                self.bias)

    def redo_backward(self):
        """Recalculate the matrix that is used for the backwards direction
        """
        logging.debug('Redo backward called')
        self.pinv = nn.Parameter(torch.pinverse(self.weight.detach(),
                                                rcond=1e-15),
                                 requires_grad=False)

    def set_backward(self, backward):
        """Set the backward synapses from the outside

        Args:
            backward (torch.tensor): Description
        """

        self.pinv = nn.Parameter(backward.float(), requires_grad=False)

    def get_forward(self):
        """Get the forward weights

        Returns:
            torch.tensor: The forward weights
        """

        return self.weight

    def get_backward(self):
        """Get the forward weights

        Returns:
            torch.tensor: The backward weights
        """

        return self.pinv
