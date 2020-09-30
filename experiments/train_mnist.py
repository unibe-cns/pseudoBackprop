"""
    An experiment to train the mnist dataset
"""
import logging
import os
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from pseudoBackprop.network import FullyConnectedNetwork

logging.basicConfig(format='Train MNIST -- %(levelname)s: %(message)s',
                    level=logging.DEBUG)


def main():
    """
        Execute the training and save the result
    """

    # parameters of the learning
    batch_size = 10
    layers = [784, 700, 10]
    epochs = 10
    model_folder = 'models'

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
    backprop_net = FullyConnectedNetwork.backprop(layers)

    # set up the optimizer and the loss function
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        backprop_net.parameters(), lr=0.001, momentum=0.9)

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
            if index % 2000 == 1999:    # print every 2000 mini-batches
                logging.info(f'epoch {epoch}, batch {index}, \
                              loss: {running_loss}')
                running_loss = 0.0

    logging.info('The training has finished')

    # save the result
    logging.info("Saving the backprop model")
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(backprop_net.state_dict(),
               os.path.join(model_folder, 'mnist_backprop.pth'))


if __name__ == '__main__':
    main()
