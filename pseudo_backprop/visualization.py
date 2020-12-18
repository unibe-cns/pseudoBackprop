"""collection of convenience function to plot the results."""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

# load the style file
dir_path = os.path.dirname(os.path.realpath(__file__))
mpl.rc_file(os.path.join(dir_path, "defaults/plotstyle"))

COLORS = {"vanilla backprop": "tab:blue",
          "feedback alignement": "tab:orange",
          "pseudo-backprop": "tab:green",
          "gen-pseudo": "tab:red"}
NAMES = ["vanilla backprop", "feedback alignement", "pseudo-backprop",
         "gen-pseudo"]


def prepare_axes(axes):
    """prepare the plot for unified plotting

    Args:
        axes: axeses object
    """
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.get_xaxis().tick_bottom()
    axes.get_yaxis().tick_left()
    axes.set_yscale('log')


# pylint: disable=R0913
def single_shot(axes, backprop=None, feedback_a=None, pseudo=None,
                gen_pseudo=None, y_type="Loss"):
    """
    Plot a single shot experiment

    Args:
        axes: an axes object
        backprop (None, optional): loss/error ratio for vanilla backprop
        fa (None, optional): loss/error ratio for feedback alignement
        pseudo (None, optional): loss/error ratio for pseudo backprop
        gen_pseudo (None, optional): loss/error ratio for gen-pseudo backprop
        type (str, optional): label on the y-axeses

    Returns:
        f: figure object
        axes: axeses object
    """
    prepare_axes(axes)

    for item, name in zip([backprop, feedback_a, pseudo, gen_pseudo], NAMES):
        if item is not None:
            axes.plot(item[:, 0], item[:, 1], linewidth=2, color=COLORS[name],
                      label=name)

    axes.legend()
    axes.set_xlabel("Epochs")
    axes.set_ylabel(y_type)


def hist_activities(axes, activities, ylabel='Frequency'):
    """make historgram of the activities

    Args:
        axes (pyplot axes object):
        activities (TYPE): Description
        ylabel (str, optional): Description
    """

    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.get_xaxis().tick_bottom()
    axes.get_yaxis().tick_left()
    axes.hist(activities)
    axes.set_xlabel('Activities')
    axes.set_ylabel(ylabel)


def plot_activities(act_arr):
    """Plot the activities in the layers

    Args:
        act_arr (list): list of activities
    """

    n_layers = len(act_arr)
    fig, axes = plt.subplots(nrows=n_layers, gridspec_kw={'hspace': 0.4})

    for index in range(n_layers):
        hist_activities(axes[index], act_arr[index].flatten(),
                        ylabel=f'Layer {index}')

    return fig
