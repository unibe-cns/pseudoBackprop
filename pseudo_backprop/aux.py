"""Standalone functions for auxillary computations."""
import numpy as np
import torch


def evaluate_model(network_model, testloader, batch_size):
    """
    Evaluate the model on the given dataset and obtain the loss function and
    the results

    Params:
        network_model: FullyConnectedNetwork object containing the neural
                       network
        testloader: the testloader object from torch
        batch_size: batch size

    Returns:
        loss: the computed loss value
        confusion_matrix: numpy matrix with the confusion matrix
    """

    confusion_matrix = np.zeros((10, 10))
    loss_function = torch.nn.CrossEntropyLoss()
    loss = 0
    # turn off gathering the gradient for testing
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.view(batch_size, -1)
            outputs = network_model(images)
            loss_value = loss_function(outputs, labels)
            loss += loss_value
            _, predicted = torch.max(outputs, 1)
            for tested in zip(labels.numpy().astype(int),
                              predicted.numpy().astype(int)):
                confusion_matrix[tested] += 1

    return loss, confusion_matrix


def generalized_pseudo(w_matrix, dataset):
    """calculate the generalized dataset

    Args:
        w_matrix (torch.tensor): forward matrix
        dataset (torch.tensor): dataset
    """

    np_dataset = dataset.numpy()
    covariance = np.cov(np_dataset)
    # make the singular value decomposition
    u_matrix, s_matrix, vh_matrix = np.linalg.svd(covariance)
    # Calculate the generalized pseudoinverse
    gamma = np.dot(np.dot(u_matrix, np.diag(np.sqrt(s_matrix))), vh_matrix)
    gen_pseudo = np.dot(gamma, np.linalg.pinv(np.dot(w_matrix, gamma)))

    return torch.from_numpy(gen_pseudo)
