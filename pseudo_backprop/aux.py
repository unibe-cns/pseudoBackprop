"""Standalone functions for auxillary computations."""
import numpy as np
import torch


def evaluate_model(network_model, testloader, batch_size, device='cpu',
                   nb_classes=10):
    """
    Evaluate the model on the given dataset and obtain the loss function and
    the results

    Params:
        network_model: FullyConnectedNetwork object containing the neural
                       network
        testloader: the testloader object from torch
        batch_size: batch size
        device (str, optional): 'gpu' or 'cpu' according to the availability

    Returns:
        loss: the computed loss value
        confusion_matrix: numpy matrix with the confusion matrix
    """
    # pylint: disable=R0914

    confusion_matrix = np.zeros((10, 10))
    loss_function = torch.nn.MSELoss(reduction='sum')
    y_onehot = torch.FloatTensor(batch_size, nb_classes)
    y_onehot.to(device)
    loss = 0
    # turn off gathering the gradient for testing
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            images = images.view(batch_size, -1)
            outputs = network_model(images)
            y_onehot.zero_()
            unsq_label = labels.unsqueeze(1)
            unsq_label.to(device)
            y_onehot.scatter_(1, unsq_label, 1)
            loss_value = loss_function(outputs, y_onehot)
            loss += loss_value
            _, predicted = torch.max(outputs, 1)
            for tested in \
                zip(labels.clone().detach().cpu().numpy().astype(int),
                    predicted.clone().detach().cpu().numpy().astype(int)):
                confusion_matrix[tested] += 1

    return loss, confusion_matrix


def generalized_pseudo(w_matrix, dataset):
    """calculate the generalized dataset

    Args:
        w_matrix (torch.tensor): forward matrix
        dataset (torch.tensor): dataset
    """

    np_dataset = dataset.detach().cpu().numpy()
    covariance = np.cov(np_dataset.T)
    # make the singular value decomposition
    u_matrix, s_matrix, vh_matrix = np.linalg.svd(covariance)
    # Calculate the generalized pseudoinverse
    gamma = np.dot(np.dot(u_matrix, np.diag(np.sqrt(s_matrix))), vh_matrix)
    gen_pseudo = np.dot(gamma, np.linalg.pinv(np.dot(w_matrix, gamma)))

    return torch.from_numpy(gen_pseudo)


def calc_loss(b_matrix, w_matrix, samples):
    """Calculate the loss based on the samples
    Args:
        b_matrix (np.ndarray): The backward matrix
        w_matrix (np.ndarray): The forward matrix
        samples : Samples to calcualte over
    Returns:
        float: the calculated loss
    """
    b_matrix = np.reshape(b_matrix, w_matrix.T.shape)
    bw_product = b_matrix.dot(w_matrix.dot(samples))
    diff = samples - bw_product
    f_samples = np.sum(np.power(diff, 2), axis=0)
    loss = np.mean(f_samples)

    return loss


def calc_activities(network, inputs, nb_layers):
    """calculate the activities throughout a network

    Args:
        network (network class): Description
        input (inout data): Description
        nb_layers (number of layers): Description
    """

    activities = []
    for layer in range(nb_layers):
        activities.append(
            network.forward_to_hidden(inputs, layer).detach().numpy())

    return activities
