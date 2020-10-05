"""An experiment to train the mnist dataset."""
import logging
import os
import argparse
import json
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from pseudo_backprop.network import FullyConnectedNetwork

logging.basicConfig(format='Train MNIST -- %(levelname)s: %(message)s',
                    level=logging.DEBUG)


def main(params):
    """
        Execute the training and save the result
    """

    # parameters of the learning
    batch_size = params["batch_size"]
    layers = params["layers"]
    epochs = params["epochs"]
    model_folder = params["model_folder"]
    model_type = params["model_type"]
    learning_rate = params["learning_rate"]
    momentum = params["momentum"]

    # set-up the folder to save the model
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # set up the normalizer
    # Normalize the images to
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(0.5, 0.5)])

    # get the dataset
    logging.info("Loading the datasets")
    trainset = torchvision.datasets.MNIST('./data', train=True,
                                          download=True,
                                          transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    logging.info("Datasets are loaded")

    # make the networks
    possible_networks = ['fa', 'backprop']
    if model_type == 'fa':
        backprop_net = FullyConnectedNetwork.feedback_alignement(layers)
    elif model_folder == 'backprop':
        backprop_net = FullyConnectedNetwork.backprop(layers)
    else:
        raise ValueError(f'{model_type} is not a valid option. Implemented \
            options are in {possible_networks}')

    # set up the optimizer and the loss function
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        backprop_net.parameters(), lr=learning_rate, momentum=momentum)

    # train the network
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        logging.info(f'Working on epoch {epoch + 1}')
        for index, data in enumerate(tqdm(trainloader), 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.view(batch_size, -1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = backprop_net(inputs)
            loss_value = loss_function(outputs, labels)
            loss_value.backward()
            optimizer.step()

            # print statistics
            # running loss is the loss measured on the last 2000 minibatches
            running_loss += loss_value.item()
            if index % (10000/batch_size) == 999:
                # print every 2000 mini-batches
                logging.info(f'epoch {epoch}, batch {index}, \
                              loss: {running_loss}')
                running_loss = 0.0
                # save the model every 10000 examples
                file_to_save = (f"model_{model_type}_epoch_{epoch}_images_"
                                f"{(index + 1) * batch_size}.pth")
                path_to_save = os.path.join(model_folder, file_to_save)
                torch.save(backprop_net.state_dict(),
                           path_to_save)

    logging.info('The training has finished')

    # save the result
    logging.info("Saving the model")

    torch.save(backprop_net.state_dict(),
               os.path.join(model_folder, 'mnist_backprop.pth'))


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='Train a model on the mnist \
        dataset.')
    PARSER.add_argument('--params', type=str,
                        help='Path to the parameter json.')
    ARGS = PARSER.parse_args()
    with open(ARGS.params, 'r+') as f:
        PARAMETERS = json.load(f)

    main(PARAMETERS)
