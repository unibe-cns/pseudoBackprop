"""Experiment to if dynamical pseudobackprop converges 
   to the data-specific pseudoinverse of the training data"""
import logging
import os
import json
import time
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from pseudo_backprop.experiments import exp_aux
from pseudo_backprop.experiments.yinyang_dataset.dataset import YinYangDataset

torch.autograd.set_detect_anomaly(True)
logging.basicConfig(format='Train model -- %(levelname)s: %(message)s',
                    level=logging.DEBUG)


# pylint: disable=R0914,R0915,R0912,R1702
def main(params, per_images=10000):

    """
        Load the training data and generate the data-specific pinverses
        Then, load the trained nets and compare the backwards weights
        with the data-specific pinverse
    """

    # parameters of the learning
    batch_size = params["batch_size"]
    layers = params["layers"]
    model_folder = params["model_folder"]
    model_type = params["model_type"]
    epochs = params["epochs"]
    if "dataset" not in params:
        dataset_type = "mnist"
    else:
        dataset_type = params["dataset"]
    if model_type != 'dyn_pseudo':
        ValueError("Invalid model type. This action can only \
                     be run for dynamical pseudobackprop")


    if dataset_type == "yinyang":
        dataset_size = params["dataset_size"]
        random_seed = params["random_seed"]

    # set random seed
    torch.manual_seed(params["random_seed"])

    # look for gpu device, use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'The training starts on device {device}.')

    # set up the normalizer
    # Normalize the images to
    transform = transforms.Compose([transforms.ToTensor()])

    # get the dataset
    logging.info("Loading the datasets")
    if dataset_type == "cifar10":
        trainset = torchvision.datasets.CIFAR10(params["dataset_path"],
                                                train=True,
                                                download=True,
                                                transform=transform)
    # yinyang is not officially implemented by torchvision, so we load it by hand:
    elif dataset_type == "yinyang":
        trainset = YinYangDataset(size = dataset_size, seed = random_seed)
        trainset.classes = trainset.class_names

    elif dataset_type == "mnist":
        trainset = torchvision.datasets.MNIST(params["dataset_path"],
                                              train=True,
                                              download=True,
                                              transform=transform)
    else:
        raise ValueError("The received dataset <<{}>> is not implemented. \
                          Choose from ['mnist', 'cifar10', 'yinyang']".format(
            dataset_type))

    nb_classes = len(trainset.classes)
    logging.info('The number of classes is %i', nb_classes)

    logging.info("Datasets are loaded")

    # make the networks
    backprop_net = exp_aux.load_network(model_type, layers)
    backprop_net.to(device)

    # make a dataloader for the training set
    genpseudo_samp = torch.utils.data.DataLoader(
            trainset,
            batch_size=len(trainset))
    genpseudo_iterator = iter(genpseudo_samp)

    sub_data = genpseudo_iterator.next()[0].view(len(trainset), -1).to(device)

    # generate a list of the data-specific pinverse matrices
    dataspecPinv_array = backprop_net.get_dataspec_pinverse(dataset=sub_data)

    logging.info("Data-specific pseudoinverse matrices calculated")

    # every <<per_images>> images there is a saved model, hence we have to
    # take into
    # account that MNIST has 60 000 images and CIFAR10 50 000
    if dataset_type == "mnist":
        nb_batches = int(60000 / per_images)
    elif dataset_type == "cifar10":
        nb_batches = int(50000 / per_images)
    elif dataset_type == "yinyang":
        per_images = 1000
        nb_batches = int(dataset_size / per_images)

    # load the saved network states and calculate Frobenius norm
    back_weights_array = []
    cos_array = []
    for index in range(epochs * nb_batches + 1):
        epoch = 0 if index == 0 else (index - 1) // nb_batches
        ims = 0 if index == 0 else (((index - 1) % nb_batches) + 1) \
            * per_images
        file_to_load = (f"model_{model_type}_epoch_{epoch}_images_"
                        f"{ims}.pth")
        logging.info(f'• Working on epoch {epoch} and image {ims}.')
        path_to_model = os.path.join(model_folder, file_to_load)
        backprop_net.load_state_dict(torch.load(path_to_model))
        # extract the backwards matrix at this stage
        back_weights_array.append(backprop_net.get_backward_weights())
        
        #logging.info(f'The backwards weight matrix is:\n {back_weights_array[-1]}')
        for layer in range(len(layers)-1):
            # calculate the cosine similarity using the Frobenius norm
            # between the data-specific pseudoinverse
            # and the dynamical backwards matrix
            cos = np.round(
                exp_aux.cosine_similarity_tensors(
                    torch.from_numpy(back_weights_array[-1][layer].T),
                    dataspecPinv_array[layer]
                    ).tolist(),
                6)
            if cos > 1 or cos < -1:
                raise ValueError(f"Cosine between tensors has returned invalid value {cos}")
            
            logging.info(f'The angle between the backwards weights and the data-specific pseudoinverse '
                                 f'in layer {layer} is: {np.arccos(cos)*180./np.pi} degrees')
            cos_array.append(cos)


   # if len(back_weights_array) != len(dataspecPinv_array):
    #    raise ValueError("Data-specific pinverse and backwards matrices of inequal length")
        



if __name__ == '__main__':

    ARGS = exp_aux.parse_experiment_arguments()
    with open(ARGS.params, 'r+') as f:
        PARAMETERS = json.load(f)

    main(PARAMETERS)
