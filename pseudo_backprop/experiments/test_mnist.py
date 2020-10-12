"""Experiment to test the trained network on the mnist dataset."""
import logging
import json
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from pseudo_backprop.aux import evaluate_model
from pseudo_backprop.experiments import exp_aux


logging.basicConfig(format='Train MNIST -- %(levelname)s: %(message)s',
                    level=logging.DEBUG)


# pylint: disable=R0914
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
    backprop_net = exp_aux.load_network(model_type, layers)

    # run over the output and evaluate the models
    loss_array = []
    conf_matrix_array = {}
    error_ratio_array = []
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
        error_ratio_array.append(1 - class_ratio)
        logging.info(f'The final classification ratio is: {class_ratio}')
        logging.info(f'The final loss function: {loss}')
        logging.info(f'The final confusion matrix is: {confusion_matrix}')

    # Save the results into an appropriate file into the model folder
    epoch_array = np.arange(0, epochs + 1/12, 1/6)
    image_array = np.arange(0, epochs * 60000 + 10000, 10000)
    to_save = np.array([epoch_array, image_array,
                        np.array(error_ratio_array), np.array(loss_array)]).T
    file_to_save = os.path.join(model_folder, f'results_{dataset}.csv')
    np.savetxt(file_to_save, to_save, delimiter=',',
               header='epochs, images, error_ratio, loss')
    with open(os.path.join(model_folder,
                           f'confusion_matrix_{dataset}.json'), 'w') as file_f:
        json.dump(conf_matrix_array, file_f, sort_keys=True, indent=4)


if __name__ == '__main__':

    ARGS = exp_aux.parse_experiment_arguments()
    with open(ARGS.params, 'r+') as f:
        PARAMETERS = json.load(f)

    main(PARAMETERS, ARGS.dataset)
