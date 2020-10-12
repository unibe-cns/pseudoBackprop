"""Plot the results of the training."""
import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pseudo_backprop import visualization as visu


def load_datafiles(path_to_json):
    """Load the datafiles based on the json description

    Args:
        path_to_json (str): path to the json file

    Returns:
        res_dict: dictionary with the results
    """
    if path_to_json is not None:
        # Load the experiment json file
        with open(path_to_json, 'r+') as datafile:
            params = json.load(datafile)

        # Load the results
        model_folder = params["model_folder"]
        file_to_load_test = os.path.join(model_folder, 'results_test.csv')
        test_res = np.loadtxt(file_to_load_test, comments='#',
                              delimiter=",")
        file_to_load_train = os.path.join(model_folder,
                                          'results_train.csv')
        train_res = np.loadtxt(file_to_load_train, comments='#',
                               delimiter=",")
        res_dict = {'train_loss': train_res[:, [0, 3]],
                    'train_error': train_res[:, [0, 2]],
                    'test_loss': test_res[:, [0, 3]],
                    'test_error': test_res[:, [0, 2]]}
    else:
        res_dict = {'train_loss': None,
                    'train_error': None,
                    'test_loss': None,
                    'test_error': None}

    return res_dict


def main(args):
    """Execute the plotting

    Args:
        args: args object from the argparser
    """
    # prepare for loading
    data = {}
    names = ['bp', 'fa', 'pseudo_bp']
    cases = [args.params_vbp, args.params_fa, args.params_pseudo]

    # Load the resutls
    for name, case in zip(names, cases):
        data[name] = load_datafiles(case)

    # make the plots
    fig, axes = plt.subplots(ncols=2, nrows=2)
    for index, mode in zip([(0, 0), (0, 1), (1, 0), (1, 1)],
                           ['train_loss', 'train_error',
                            'test_loss', 'test_error']):
        y_label = 'Loss' if 'loss' in mode else 'Error ratio'
        visu.single_shot(axes[index], feedback_a=data['fa'][mode],
                         backprop=data['bp'][mode],
                         pseudo=data['pseudo_bp'][mode],
                         y_type=y_label)

    fig.savefig('results.png')


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='Plot the results on\
                                                  the MNIST dataset.')
    PARSER.add_argument('--params_vbp', type=str, default=None,
                        help='Path to the vanilla backprop parameter json.')
    PARSER.add_argument('--params_fa', type=str, default=None,
                        help='Path to the feedback alignement parameter json.')
    PARSER.add_argument('--params_pseudo', type=str, default=None,
                        help='Path to the pseudo backprop parameter json.')
    ARGS = PARSER.parse_args()

    main(ARGS)
