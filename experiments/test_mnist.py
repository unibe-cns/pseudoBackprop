"""Experiment to test the trained network on the mnist dataset."""
import logging
import argparse
import json
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from pseudo_backprop.network import FullyConnectedNetwork
from pseudo_backprop.aux import evaluate_model

logging.basicConfig(format='Train MNIST -- %(levelname)s: %(message)s',
                    level=logging.DEBUG)


def main(params, dataset):
    """Run the testing on the mnist dataset."""
    # The metaparameter
    layers = params['layers']
    batch_size = params['batch_size']
    model_type = params['model_type']
    model_folder = params["model_folder"]
    epochs = params["epochs"]

    # Load the model and the data
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(0.5, 0.5)])
    testset = torchvision.datasets.MNIST('./data', train=(dataset == 'train'),
                                         download=True,
                                         transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=True, num_workers=2)

    # make the networks
    possible_networks = ['fa', 'backprop']
    if model_type == 'fa':
        backprop_net = FullyConnectedNetwork.feedback_alignement(layers)
    elif model_type == 'backprop':
        backprop_net = FullyConnectedNetwork.backprop(layers)
    else:
        raise ValueError(f'{model_type} is not a valid option. Implemented \
            options are in {possible_networks}')

    # run over the output and evaluate the models
    loss_array = []
    conf_matrix_array = {}
    class_ratio_array = []
    for index in range(epochs * 6 + 1):
        epoch = 0 if index == 0 else (index - 1) // 6
        ims = 0 if index == 0 else (((index - 1) % 6) + 1) * 10000
        file_to_load = (f"model_{model_type}_epoch_{epoch}_images_"
                        f"{ims}.pth")
        logging.info(f'Working on epoch {epoch} and image {ims}.')
        path_to_model = os.path.join(model_folder, file_to_load)
        backprop_net.load_state_dict(torch.load(path_to_model))
        # Evaluate the model
        loss, confusion_matrix = evaluate_model(backprop_net, testloader,
                                                batch_size)
        class_ratio = (confusion_matrix.diagonal().sum() /
                       confusion_matrix.sum())
        loss_array.append(loss)
        conf_matrix_array[index] = confusion_matrix.tolist()
        class_ratio_array.append(class_ratio)
        logging.info(f'The final classification ratio is: {class_ratio}')
        logging.info(f'The final loss function: {loss}')
        logging.info(f'The final confusion matrix is: {confusion_matrix}')

    # Save the results into an appropriate file into the model folder
    epoch_array = np.arange(0, epochs, 1/6)
    image_array = np.arange(0, epochs * 60000 + 10000, 10000)
    to_save = np.concatenate([epoch_array, image_array,
                              np.array(class_ratio_array)]).T
    file_to_save = os.path.join(model_folder, f'results_{dataset}.csv')
    np.savetxt(file_to_save, to_save, delimiter=',',
               header='epochs, images, class_ratio')
    with open(os.path.join(model_folder,
                           f'confusion_matrix_{dataset}.json'), 'w') as file_f:
        json.dump(conf_matrix_array, file_f, sort_keys=True, indent=4)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='Train a model on the mnist \
        dataset.')
    PARSER.add_argument('--params', type=str,
                        help='Path to the parameter json.')
    PARSER.add_argument('--dataset', type=str,
                        help='Choose from <test> or <train>.',
                        default='test')
    ARGS = PARSER.parse_args()
    with open(ARGS.params, 'r+') as f:
        PARAMETERS = json.load(f)

    main(PARAMETERS, ARGS.dataset)
