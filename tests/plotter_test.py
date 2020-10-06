"""Test the visualization of the package."""
import numpy as np
from pseudo_backprop import visualization as visu


# Set up random data to create the plots
X_ARRAY = np.linspace(0, 100, 100)
Y_ARRAY_1 = np.random.normal(size=100)
Y_ARRAY_2 = np.random.normal(size=100) + 3
Y_ARRAY_3 = np.random.normal(size=100) + 6


def single_plot_test():
    """Make a single plot"""

    fig, _ = visu.single_shot(X_ARRAY, backprop=Y_ARRAY_1,
                              feedback_a=Y_ARRAY_2, pseudo=Y_ARRAY_3)
    fig.savefig("testplot.png")
