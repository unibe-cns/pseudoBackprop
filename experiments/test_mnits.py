"""Experiment to test the trained network on the mnist dataset."""
import logging
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from pseudo_backprop.network import FullyConnectedNetwork

logging.basicConfig(format='Train MNIST -- %(levelname)s: %(message)s',
                    level=logging.DEBUG)


def main():
    """Run the testing on the mnist dataset."""
    # The metaparameter
    path_to_model = 'models/mnist_backprop.pth'
    layers = [784, 700, 10]
    batch_size = 10

    # Load the model and the data
    backprop_net = FullyConnectedNetwork.backprop(layers)
    backprop_net.load_state_dict(torch.load(path_to_model))
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(0.5, 0.5)])
    testset = torchvision.datasets.MNIST('./data', train=False,
                                         download=True,
                                         transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=True, num_workers=2)

    # go over the set and get the results per class
    # check for the single classes
    logging.info("Working on the pred accuracy for single classes")
    confusion_matrix = np.zeros((10, 10))
    # turn off gathering the gradient for testing
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.view(batch_size, -1)
            outputs = backprop_net(images)
            _, predicted = torch.max(outputs, 1)
            for tested in zip(labels.numpy().astype(int),
                              predicted.numpy().astype(int)):
                confusion_matrix[tested] += 1

    # report the final classification ratio
    class_ratio = confusion_matrix.diagonal().sum() / confusion_matrix.sum()
    logging.info(f'The final classification ratio is: {class_ratio}')


if __name__ == '__main__':
    main()
