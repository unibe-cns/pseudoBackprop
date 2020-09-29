"""
    Modules that define the backward path for the feedback alignment
    and the pseudo-prop lienarities
"""
import torch
from torch import nn


class FeedbackAlignmentLinearity(torch.autograd.Function):
    """
        The feedack alignment function
        This defines the forward and the backwords directions
    """

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
        output = torch.matmul(input_torch, weight.t())
        if bias is not None:
            output += torch.unsqueeze(bias, 0).expand_as(output)

        return output

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

        print(ctx.needs_input_grad)
        # calculate the gradients
        # if ctx.needs_input_grad[0]:
        # gradient at the input of the forward pass
        grad_input = torch.matmul(grad_output, back_weight.t())
        # if ctx.needs_input_grad[1]:
        # gradient at the weights
        grad_weight = torch.matmul(grad_output.t(), input_torch)
        if bias is not None and ctx.needs_input_grad[3]:
            # gradient at the bias if required
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_back_weight, grad_bias


class FeedbackAlginementModule(nn.Module):

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

        # create the parameters
        self.w_matrix = nn.Parameter(torch.Tensor(self.output_size,
                                                  self.input_size),
                                     requires_grad=True)
        # create the biases if applicable
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_size),
                                     requires_grad=True)
        else:
            self.register_parameters('bias', None)

        # create a variable for the random feedback weights
        self.w_back = torch.autograd.Variable(
                                torch.FloatTensor(self.input_size,
                                                  self.output_size),
                                requires_grad=False)

        # Initialize the weights
        torch.nn.init.kaiming_normal_(self.w_matrix, mode='fan_in',
                                      nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.w_back, mode='fan_in',
                                      nonlinearity='leaky_relu')
        torch.nn.init.normal_(self.bias)

    def forward(self, input_tensor):
        # the forward calcualtion of the module
        return FeedbackAlignmentLinearity.apply(input_tensor,
                                                self.w_matrix,
                                                self.w_back,
                                                self.bias)
