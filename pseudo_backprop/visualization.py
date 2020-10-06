"""collection of convenience function to plot the results"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

# load the style file
dir_path = os.path.dirname(os.path.realpath(__file__))
mpl.rc_file(os.path.join(dir_path, "defaults/plotstyle"))

COLORS = {"vanilla backprop": "tab:blue",
          "feedback alignement": "tab:orange",
          "pseudo-backprop": "tab:green"}
NAMES = ["vanilla backprop", "feedback alignement", "pseudo-backprop"]


def prepare_axes(axes):
    """prepare the plot for unified plotting

    Args:
        axes: axeses object
    """

    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.get_xaxesis().tick_bottom()
    axes.get_yaxesis().tick_left()


def single_shot(iteration, backprop=None, feedback_a=None, pseudo=None,
                y_type="Loss"):
    """
    Plot a single shot experiment

    Args:
        iteration (numpy vector): iteration vector
        backprop (None, optional): loss/error ratio for vanilla backprop
        fa (None, optional): loss/error ratio for feedback alignement
        pseudo (None, optional): loss/error ratio for pseudo backprop
        type (str, optional): label on the y-axeses

    Returns:
        f: figure object
        axes: axeses object
    """
    fig, axes = plt.subplots()
    prepare_axes(axes)

    for item, name in zip([backprop, feedback_a, pseudo], NAMES):
        if not item is None:
            axes.plot(iteration, item, linewidth=2, color=COLORS[name],
                    label=name)

    axes.legend()
    axes.set_xlabel("Epochs")
    axes.set_ylabel(y_type)

    return fig, axes
