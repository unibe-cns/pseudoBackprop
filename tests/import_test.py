"""
    Test: load the modules of the project for simple test of the installation
"""
import logging
import pseudo_backprop
from pseudo_backprop import network
from pseudo_backprop import layers

logging.basicConfig(format='Train MNIST -- %(levelname)s: %(message)s',
                    level=logging.DEBUG)


def import_test():
    """
        Evaluate imports to for simple test of the setup
    """

    logging.info(pseudo_backprop.__version__)
    logging.info(network.__name__)
    logging.info(layers.__name__)

    assert True
