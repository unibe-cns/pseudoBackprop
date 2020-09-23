"""
    Modules that define the backward path for the feedback alignment
    and the pseudo-prop lienarities
"""
import torch


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
            input: the input tensor
            weight: the forward weight matrix
            back_weight: the backward weight matrix which is fix during learing
            bias: tensor of the bias variables if applicable
        """

        ctx.save_for_backward(input_torch, weight, back_weight, bias)
        output = torch.matmul(weight, input_torch)
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

        # calculate the gradients
        if ctx.needs_input_grad[0]:
            # gradient at the input of the forward pass
            grad_input = torch.matmul(back_weight, grad_output)
        if ctx.needs_input_grad[1]:
            # gradient at the weights
            grad_weight = torch.matmul(grad_output.t(), input_torch)
        if bias is not None and ctx.needs_input_grad[3]:
            # gradient at the bias if required
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_back_weight, grad_bias
