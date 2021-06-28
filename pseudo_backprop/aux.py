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

    confusion_matrix = np.zeros((nb_classes, nb_classes))
    loss_function = torch.nn.MSELoss(reduction='sum')
    y_onehot = torch.empty(batch_size, nb_classes, device=device)
    y_onehot.to(device)
    loss = 0
    # turn off gathering the gradient for testing
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            images = images.float()     # for yinyang, we need to convert to float32
            images = images.view(batch_size, -1)
            outputs = network_model(images)
            y_onehot.zero_()
            unsq_label = labels.unsqueeze(1)
            y_onehot.to(device)
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

def loss_error(network_model, testloader, batch_size, device='cpu',
                   nb_classes=10):
    """
    Evaluate the model on the given dataset and return the error vector

    Params:
        network_model: FullyConnectedNetwork object containing the neural
                       network
        testloader: the testloader object from torch
        batch_size: batch size
        device (str, optional): 'gpu' or 'cpu' according to the availability

    Returns:
        error: (target - output) vector of shape (output_neurons,)
               averaged over whole batch
               as torch array
    """
    # pylint: disable=R0914

    y_onehot = torch.empty(batch_size, nb_classes, device=device)
    y_onehot.to(device)
    error = torch.zeros(batch_size, nb_classes, device=device)
    error.to(device)
    # turn off gathering the gradient for testing
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            images = images.float()     # for yinyang, we need to convert to float32
            images = images.view(batch_size, -1)
            outputs = network_model(images)
            y_onehot.zero_()
            unsq_label = labels.unsqueeze(1)
            y_onehot.to(device)
            unsq_label.to(device)
            y_onehot.scatter_(1, unsq_label, 1)
            error += (y_onehot - outputs)

    # sum over batch
    error = torch.sum(error, dim=0)
    # divide by number of samples to get averaged error
    error /= (len(testloader) * batch_size)

    return error

def torchcov(m, rowvar=True, inplace=False):
    # implements covariance estimator 
    # via https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/4

    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    if inplace:
        m -= torch.mean(m, dim=1, keepdim=True)
    else:
        m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()

def generalized_pseudo(w_matrix, dataset):
    """calculate the dataspecific pseudoinverse

    Args:
        w_matrix (torch.tensor): forward matrix
        dataset (torch.tensor): dataset
    """

    np_dataset = dataset.detach().cpu().numpy()
    covariance = np.cov(np_dataset.T)
    mean = np.mean(np_dataset, axis=0)
    gammasquared = covariance + np.outer(mean,mean)
    
    # make the singular value decomposition
    u_matrix, s_matrix, vh_matrix = np.linalg.svd(gammasquared)
    # Calculate the generalized pseudoinverse
    gamma = np.dot(np.dot(u_matrix, np.diag(np.sqrt(s_matrix))), vh_matrix)
    gen_pseudo = np.dot(gamma, np.linalg.pinv(np.dot(w_matrix, gamma)))

    return torch.from_numpy(gen_pseudo)


def calc_gamma_matrix(dataset):
    """calculate square root of the
       autocorrelation Gamma^2 = <rr^T>

    Args:
        dataset (torch.tensor): dataset r
        (tensor of data vectors r)
    """

    np_dataset = dataset.detach().cpu().numpy()
    covariance = np.cov(np_dataset.T)
    mean = np.mean(np_dataset, axis=0)
    gammasquared = covariance + np.outer(mean,mean)
    # print('mean: ', mean)
    # print('cov:', covariance)
    
    # make the singular value decomposition
    u_matrix, s_matrix, vh_matrix = np.linalg.svd(gammasquared)
    # Calculate the generalized pseudoinverse
    gamma = np.dot(np.dot(u_matrix, np.diag(np.sqrt(s_matrix))), vh_matrix)

    return torch.from_numpy(gamma)

def calc_gamma2_matrix(dataset):
    """calculate
       autocorrelation Gamma^2 = <rr^T>

    Args:
        dataset (torch.tensor): dataset r
        (tensor of data vectors r)
    """

    np_dataset = dataset.detach().cpu().numpy()
    covariance = np.cov(np_dataset.T)
    mean = np.mean(np_dataset, axis=0)
    gammasquared = covariance + np.outer(mean,mean)

    return torch.from_numpy(gammasquared)

def calc_gamma2_matrix_torch(dataset):
    """calculate
       autocorrelation Gamma^2 = <rr^T>
       in torch. This is faster than numpy

    Args:
        dataset (torch.tensor): dataset r
        (tensor of data vectors r)
    """

    covariance = torchcov(dataset.T)
    mean = torch.mean(dataset,axis=0)

    return covariance + torch.outer(mean,mean)


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

def calc_mismatch_energy(Gamma, B, W, alpha):
    """calculates the mismatch energy between B
       and the data-specific pseudoinverse of W

    Args:
        Gamma: square root of data vector
        B: backwards matrix
        W: forwards matrix

        all as numpy arrays
    """

    mismatch_energy = .5 * np.linalg.norm(Gamma - B @ W @ Gamma)**2 + alpha/2. * np.linalg.norm(B)**2

    return mismatch_energy

def calc_mismatch_energy_fast(Gamma2, B, W, alpha):
    """calculates the mismatch energy between B
       and the data-specific pseudoinverse of W
       using Gamma squared

    Args:
        Gamma2: square root of data vector
        B: backwards matrix
        W: forwards matrix

        all as numpy arrays
    """

    I = np.identity(np.shape(Gamma2)[0])

    mismatch_energy = .5 * np.trace((I - B @ W).T @ (I - B @ W) @ Gamma2) + alpha/2. * np.linalg.norm(B)**2

    return mismatch_energy
