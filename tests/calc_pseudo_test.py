"""Test the setup and evaluation of the networks."""
import logging
import torch
import nose
import numpy as np
from pseudo_backprop.network import FullyConnectedNetwork

logging.basicConfig(format='Tests -- %(levelname)s: %(message)s',
                    level=logging.DEBUG)


class TestClassPseudoBackpropagation:
    """
        test the setup of a pseudo backpropagation network with one forward and
        one backwar step on synthetic data
    """

    @classmethod
    def setup_class(cls):
        """Set up an architecture and create a network!"""
        cls.layers = [20, 30, 32, 30, 10]
        cls.net = FullyConnectedNetwork.pseudo_backprop(cls.layers)
        cls.random_input = torch.randn(1, 20)
        cls.random_output = torch.randn(1, 10)
        cls.net.redo_backward_weights()

    def calc_pseudo_test_init(self):
        """ make the network calculate the pseudo-inverse and check if it is
            true in the beginning
        """
        logging.info('Running the initial pseudo test.')

        forward_weights = self.net.get_forward_weights()
        backward_weights = self.net.get_backward_weights()
        print(forward_weights)
        sizes_forward = [w.shape for w in forward_weights]
        sizes_backward = [w.shape for w in backward_weights]
        print(sizes_forward)
        print(sizes_backward)
        frob_norms = [np.linalg.norm(np.linalg.pinv(w)-b, ord='fro') for (w, b)
                      in zip(forward_weights, backward_weights)]
        frob_norm_rand = [np.linalg.norm(
            np.linalg.pinv(w)-np.random.rand(*b.shape),
            ord='fro') for (w, b)
            in zip(forward_weights, backward_weights)]
        print(frob_norms)
        print(frob_norm_rand)
        for (frob, frob_rand) in zip(frob_norms, frob_norm_rand):
            nose.tools.assert_greater(frob_rand, frob)

    def calc_pseudo_test_train(self):
        """ make the network calculate the pseudo-inverse and check if it is
            true after some training """
        logging.info('Running pseudo test after training')

        optimizer = torch.optim.SGD(self.net.parameters(),
                                    lr=0.001, momentum=0.6)
        loss_function = torch.nn.MSELoss()

        # make some training steps
        for _ in range(500):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.net(self.random_input)
            loss_value = loss_function(outputs, self.random_output)
            loss_value.backward()
            optimizer.step()

        forward_weights = self.net.get_forward_weights()
        backward_weights = self.net.get_backward_weights()
        frob_norms_before = [np.linalg.norm(np.linalg.pinv(w)-b, ord='fro')
                             for (w, b)
                             in zip(forward_weights, backward_weights)]
        self.net.redo_backward_weights()
        backward_weights = self.net.get_backward_weights()
        frob_norms_after = [np.linalg.norm(
            np.linalg.pinv(w)-b,
            ord='fro') for (w, b)
            in zip(forward_weights, backward_weights)]
        print(frob_norms_before)
        print(frob_norms_after)
        for (frob_bef, frob_af) in zip(frob_norms_before, frob_norms_after):
            nose.tools.assert_greater(frob_bef, frob_af)


class TestClassGenPseudoBackpropagation:
    """
        test the setup of a pseudo backpropagation network with one forward and
        one backwar step on synthetic data
    """

    @classmethod
    def setup_class(cls):
        """Set up an architecture and create a network!"""
        cls.layers = [20, 30, 32, 30, 10]
        cls.net = FullyConnectedNetwork.gen_pseudo_backprop(cls.layers)
        cls.random_input = torch.randn(200, 20)
        cls.random_output = torch.randn(200, 10)
        cls.net.redo_backward_weights(cls.random_input)

    def calc_pseudo_test_init(self):
        """ make the network calculate the pseudo-inverse and check if it is
            true in the beginning
        """
        logging.info('Running the initial pseudo test.')

        forward_weights = self.net.get_forward_weights()
        backward_weights = self.net.get_backward_weights()
        print(forward_weights)
        sizes_forward = [w.shape for w in forward_weights]
        sizes_backward = [w.shape for w in backward_weights]
        print(sizes_forward)
        print(sizes_backward)
        frob_norms = [np.linalg.norm(np.linalg.pinv(w)-b, ord='fro') for (w, b)
                      in zip(forward_weights, backward_weights)]
        frob_norm_rand = [np.linalg.norm(
            np.linalg.pinv(w)-np.random.rand(*b.shape),
            ord='fro') for (w, b)
            in zip(forward_weights, backward_weights)]
        print(frob_norms)
        print(frob_norm_rand)
        # for (frob, frob_rand) in zip(frob_norms, frob_norm_rand):
        #    nose.tools.assert_greater(frob_rand, frob)

    def calc_pseudo_test_train(self):
        """ make the network calculate the pseudo-inverse and check if it is
            true after some training """
        logging.info('Running pseudo test after training')

        optimizer = torch.optim.SGD(self.net.parameters(),
                                    lr=0.001, momentum=0.6)
        loss_function = torch.nn.MSELoss()

        # make some training steps
        for _ in range(500):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.net(self.random_input)
            loss_value = loss_function(outputs, self.random_output)
            loss_value.backward()
            optimizer.step()

        forward_weights = self.net.get_forward_weights()
        backward_weights = self.net.get_backward_weights()
        frob_norms_before = [np.linalg.norm(np.linalg.pinv(w)-b, ord='fro')
                             for (w, b)
                             in zip(forward_weights, backward_weights)]
        self.net.redo_backward_weights(self.random_input)
        backward_weights = self.net.get_backward_weights()
        frob_norms_after = [np.linalg.norm(
            np.linalg.pinv(w)-b,
            ord='fro') for (w, b)
            in zip(forward_weights, backward_weights)]
        print(frob_norms_before)
        print(frob_norms_after)
        # for (frob_bef, frob_af) in zip(frob_norms_before, frob_norms_after):
        #    nose.tools.assert_greater(frob_bef, frob_af)


# testClass = TestClassGenPseudoBackpropagation()
# testClass.setup_class()
# testClass.calc_pseudo_test_init()
