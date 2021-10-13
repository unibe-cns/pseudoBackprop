"""Experiment to test the trained network on the mnist dataset."""
import logging
import json
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from pseudo_backprop.aux import evaluate_model
from pseudo_backprop.aux import generalized_pseudo
from pseudo_backprop.experiments import exp_aux
from pseudo_backprop.experiments.yinyang_dataset.dataset import YinYangDataset
from pseudo_backprop.experiments.parity_dataset.dataset import ParityDataset


logging.basicConfig(format='Test model -- %(levelname)s: %(message)s',
                    level=logging.DEBUG)


# pylint: disable=R0914,R0915
def main(params, dataset, per_images, num_workers):
    """
    Run the testing on the mnist dataset.
    
    If training has failed (e.g. because of runaway weight matrices)
    files are only written up to the last valid model snapshot 
   
    """

    # The metaparameter
    layers = params['layers']
    batch_size = 25  # for training this is optimized for speed
    model_type = params['model_type']
    model_folder = params["model_folder"]
    epochs = params["epochs"]
    if "dataset" not in params:
        dataset_type = "mnist"
    else:
        dataset_type = params["dataset"]
    if "bias" in params:
            bias = params["bias"]
    else:
        bias = True

    if dataset_type in ["yinyang", "parity"]:
        dataset_size = params["dataset_size"]
    random_seed = params["random_seed"]
    if "criterion" not in params:
        loss_criterion = "MSELoss"
    else:
        loss_criterion = params["criterion"]

    # set random seed
    torch.manual_seed(random_seed)

    # set width of terminal output for numpy arrays
    np.set_printoptions(linewidth=160)

    # look for device, use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'The training starts on device {device}.')

    # Load the model and the data
    transform = transforms.Compose([transforms.ToTensor()])
    if dataset_type == "cifar10":
        if per_images == None: per_images = 10000
        testset = torchvision.datasets.CIFAR10(params["dataset_path"],
                                               train=(dataset == 'train'),
                                               download=True,
                                               transform=transform)
    # yinyang is not officially implemented by torchvision, so we load it by hand:
    elif dataset_type == "yinyang":
        if per_images == None: per_images = 1000
        if dataset == 'test':
            testset = YinYangDataset(size = 1000, seed = random_seed + 1)
        else:
            testset = YinYangDataset(size = dataset_size, seed = random_seed)
        testset.classes = testset.class_names
        # implemntation of parity dataset:
    elif dataset_type == "parity":
        testset = ParityDataset(inputs = layers[0], samples=dataset_size, seed = random_seed)
        batch_size = params["batch_size"]
        testset.classes = testset.class_names

    elif dataset_type == "mnist":
        if per_images == None: per_images = 10000
        testset = torchvision.datasets.MNIST(params["dataset_path"],
                                             train=(dataset == 'train'),
                                             download=True,
                                             transform=transform)
    else:
        raise ValueError("The received dataset <<{}>> is not implemented. \
                          Choose from ['mnist', 'cifar10', 'yinyang', 'parity']".format(dataset_type))
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=True, num_workers=num_workers)
    nb_classes = len(testset.classes)

    # make the networks
    net_params =   {"bias" : bias}
    backprop_net = exp_aux.load_network(model_type, 
                                        layers,
                                        net_params)
    backprop_net.to(device)

    if per_images is None:
        if "per_images" in params:
            per_images = params["per_images"]
        else:
            # define how often we shall print and output
            if dataset_type == "yinyang": per_images = dataset_size // 10
            elif dataset_type == "parity": per_images = dataset_size // 2
            else: per_images = 10000

    # every <<per_images>> images there is a saved model, hence we have to
    # take into
    # account that MNIST has 60 000 images and CIFAR10 50 000
    if dataset_type == "mnist":
        nb_batches = int(60000 / per_images)
    elif dataset_type == "cifar10":
        nb_batches = int(50000 / per_images)
    elif dataset_type == "yinyang":
        nb_batches = int(dataset_size / per_images)
    elif dataset_type == "parity":
        nb_batches = int(dataset_size / per_images)

    # run over the output and evaluate the models
    loss_array = []
    conf_matrix_array = {}
    error_ratio_array = []
    norm_forward_weight_array = []
    norm_back_weight_array = []
    epoch_array = []
    image_array = []
    for index in range(epochs * nb_batches + 1):
        epoch = 0 if index == 0 else (index - 1) // nb_batches
        ims = 0 if index == 0 else (((index - 1) % nb_batches) + 1) \
            * per_images
        file_to_load = (f"model_{model_type}_epoch_{epoch}_images_"
                        f"{ims}.pth")
        logging.info(f'• Processing model at state of epoch {epoch} and image {ims}.')
        path_to_model = os.path.join(model_folder, file_to_load)
        try:
            backprop_net.load_state_dict(torch.load(path_to_model))
        except FileNotFoundError:
            # Save a final line with nan values
            logging.info(f'File not found. Check that model has trained successfully.')
            epoch_array.append(epoch)
            image_array.append(ims)
            loss_array.append(np.nan)
            error_ratio_array.append(np.nan)
            norm_forward_weight_array.append(
                    np.arange(len(backprop_net.synapses)) * np.nan
                    )
            norm_back_weight_array.append(
                    np.arange(len(backprop_net.synapses)) * np.nan
                    )
            break

        # Evaluate the model
        loss, confusion_matrix = evaluate_model(backprop_net, testloader,
                                                batch_size, device,
                                                nb_classes, loss_criterion)
        class_ratio = (confusion_matrix.diagonal().sum() /
                       confusion_matrix.sum())
        loss_array.append(loss)
        conf_matrix_array[index] = confusion_matrix.tolist()
        error_ratio_array.append(1 - class_ratio)

        if model_type == 'dyn_pseudo':

            norm_forward_weight = []
            norm_back_weight = []
            for i in range(len(backprop_net.synapses)):
                       norm_forward_weight.append(np.linalg.norm(backprop_net.synapses[i].get_forward().cpu()))
                       # logging.info(f'The Frobenius norm of the forward weights in layer {i} is: {norm_forward_weight[-1]}')
                       norm_back_weight.append(np.linalg.norm(backprop_net.synapses[i].get_backward().cpu()))
                       # logging.info(f'The Frobenius norm of the backward weights in layer {i} is: {norm_back_weight[-1]}')

            norm_forward_weight_array.append(np.array(norm_forward_weight.copy()).T)
            norm_back_weight_array.append(np.array(norm_back_weight.copy()).T)

        logging.info(f'The final classification ratio is: {class_ratio}')
        logging.info(f'The final loss function: {loss}')
        logging.info(f'The final confusion matrix is:\n {confusion_matrix}')

        epoch_array.append(epoch)
        image_array.append(ims)

    # Save the results into an appropriate file into the model folder
    to_save = np.array([epoch_array, image_array,
                        np.array(error_ratio_array), np.array(loss_array)]).T
    file_to_save = os.path.join(model_folder, f'results_{dataset}.csv')
    np.savetxt(file_to_save, to_save, delimiter=',', fmt='%i, %i, %1.4f, %1.4f',
               header='epochs, images, error_ratio, loss')
    with open(os.path.join(model_folder,
                           f'confusion_matrix_{dataset}.json'), 'w') as file_f:
        json.dump(conf_matrix_array, file_f, sort_keys=True, indent=4)

    if model_type == 'dyn_pseudo':
        # convert array of weight matrices to numpy and save
        # norm_forward_weight_array = np.array(norm_forward_weight_array)
        # norm_back_weight_array = np.array(norm_back_weight_array)
        # to_save_matrix_norms = np.array(norm_forward_weight_array, norm_back_weight_array]).T

        layer_names = [str(i) for i in list(range(len(norm_forward_weight_array[-1])))]

        file_to_save_fw_norms = os.path.join(model_folder, f'forward_norms_{dataset}.csv')
        to_save = np.array([epoch_array, image_array])
        to_save = np.append(to_save, np.array(norm_forward_weight_array).T, axis=0).T
        np.savetxt(file_to_save_fw_norms, to_save, delimiter=',',
                       header='epochs, images, ' + 'layer ' + ' ,'.join([layer for layer in layer_names]))

        file_to_save_bw_norms = os.path.join(model_folder, f'backwards_norms_{dataset}.csv')
        to_save = np.array([epoch_array, image_array])
        to_save = np.append(to_save, np.array(norm_back_weight_array).T, axis=0).T
        np.savetxt(file_to_save_bw_norms, to_save, delimiter=',',
                       header='epochs, images, ' + 'layer ' + ' ,'.join([layer for layer in layer_names]))

if __name__ == '__main__':

    ARGS = exp_aux.parse_experiment_arguments()
    with open(ARGS.params, 'r+') as f:
        PARAMETERS = json.load(f)

    main(PARAMETERS, ARGS.dataset, per_images=ARGS.per_images, num_workers=ARGS.num_workers)
