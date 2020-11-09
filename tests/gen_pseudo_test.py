"""
    Test the generalized pseudo in pytorch
"""
import torch
import nose
import numpy as np
from pseudo_backprop import aux


def gen_pseudo_test():
    """
        Test the gen pseudo

        The test is based on the fect that for W in R^(d_out, d_in)
        and d_out > d_in, the gen_pseudo should fall back to the normal pseudo
        inverse.
    """
    d_in = 3
    d_out = 5
    n_samples = 400

    w_matrix = torch.rand(d_out, d_in)
    dataset = torch.rand(n_samples, d_in)
    gen_pseudo = aux.generalized_pseudo(w_matrix, dataset)

    # make the test
    pseudo_inv = torch.pinverse(w_matrix)

    deviation = np.abs(pseudo_inv - gen_pseudo) < 1e-3
    nose.tools.ok_(deviation.all())
