"""Test the setup and evaluation of the networks."""
import logging
import torch
import nose
import numpy as np
from pseudo_backprop.network import FullyConnectedNetwork
import pseudo_backprop.aux as aux

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
        """ Calculate the pseudo-inverse in network and check (init)
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
        """ Calculate the pseudo-inverse in network and check (trained) """
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
        """ Calculate the pseudo-inverse in network and check (init)
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
        self._check_gen_pseudo()

    def calc_pseudo_test_train(self):
        """ Calculate the pseudo-inverse in network and check (trained) """
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
            print(loss_value)
            optimizer.step()

        self.net.redo_backward_weights(self.random_input)
        self._check_gen_pseudo()

    def _check_gen_pseudo(self):
        """Go over the synapses and check them for the gen pseudoinverse
        """

        forward_weights = self.net.get_forward_weights()
        backward_weights = self.net.get_backward_weights()

        for index in range(len(self.layers) - 1):
            if self.layers[index] <= self.layers[index + 1]:
                logging.info('Testing against pseudoinverse')
                diff = np.linalg.pinv(
                    forward_weights[index]) - backward_weights[index]
                frob_norm = np.linalg.norm(diff, ord='fro')
                frob_rand = np.linalg.norm(
                    backward_weights[index] -
                    np.random.rand(*backward_weights[index].shape), ord='fro')
                logging.info(f'Frob norm: B to random {frob_rand}')
                logging.info(f'Frob norm: B to W-pseudo {frob_norm}')
                nose.tools.assert_greater(frob_rand, frob_norm)
            else:
                logging.info('Testing against energy')
                w_matrix = forward_weights[index]
                b_matrix = backward_weights[index]
                input_data = self.net.forward_to_hidden(self.random_input,
                                                        index)
                loss_b = aux.calc_loss(b_matrix,
                                       w_matrix,
                                       input_data.detach().numpy().T)
                loss_pinv = aux.calc_loss(np.linalg.pinv(w_matrix),
                                          w_matrix,
                                          input_data.detach().numpy().T)
                loss_random = aux.calc_loss(np.random.rand(*b_matrix.shape),
                                            w_matrix,
                                            input_data.detach().numpy().T)
                logging.info(f"Loss with gen pseudo: {loss_b}")
                logging.info(f"Loss with W pinv: {loss_pinv}")
                logging.info(f"Loss with random matrix: {loss_random}")
                nose.tools.assert_greater(loss_random, loss_pinv)
                nose.tools.assert_greater(loss_pinv, loss_b)
