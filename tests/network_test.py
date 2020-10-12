"""Test the setup and evaluation of the networks."""
import torch
from pseudo_backprop.network import FullyConnectedNetwork


class TestClassBackprop:
    """
        test the setup of a backprop network with one forward and
        one backwar step on synthetic data
    """

    @classmethod
    def setup_class(cls):
        """Set up an architecture and create a network!"""
        cls.layers = [200, 300, 320, 300, 10]
        cls.backprop_net = FullyConnectedNetwork.backprop(cls.layers)
        cls.random_input = torch.randn(1, 200)
        cls.random_output = torch.randn(1, 10)

    def forward_bpnetwork_test(self):
        """Make a forward pass through the network!"""
        self.backprop_net(self.random_input)

    def backward_bpnetwork_test(self):
        """Make a backward pass to calculate the gradients!"""
        out = self.backprop_net(self.random_input)
        self.backprop_net.zero_grad()
        out.backward(self.random_output)


class TestClassFeedbackAlignement:
    """
        test the setup of a feeadback alignment network with one forward and
        one backwar step on synthetic data
    """

    @classmethod
    def setup_class(cls):
        """Set up an architecture and create a network!"""
        cls.layers = [200, 300, 320, 300, 10]
        cls.fa_net = FullyConnectedNetwork.feedback_alignement(cls.layers)
        cls.random_input = torch.randn(1, 200)
        cls.random_output = torch.randn(1, 10)

    def forward_feedbackalignement_test(self):
        """Make a forward pass through the network!"""
        self.fa_net(self.random_input)

    def backward_feedbackalignement_test(self):
        """Make a backward pass to calculate the gradients!"""
        out = self.fa_net(self.random_input)
        self.fa_net.zero_grad()
        out.backward(self.random_output)
