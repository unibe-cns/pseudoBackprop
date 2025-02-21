"""An experiment to train the mnist dataset."""
import logging
import os
import json
import time
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from pseudo_backprop.experiments import exp_aux
from pseudo_backprop.experiments.yinyang_dataset.dataset import YinYangDataset
from pseudo_backprop.experiments.parity_dataset.dataset import ParityDataset
from pseudo_backprop.aux import *

#torch.autograd.set_detect_anomaly(True)
logging.basicConfig(format='Train model -- %(levelname)s: %(message)s',
                    level=logging.DEBUG)

PRINT_DEBUG = False

# pylint: disable=R0914,R0915,R0912,R1702
def main(params, per_images, num_workers):

    # time of initiation, used for timing
    t0 = time.time()

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
    if "bias" in params:
            bias = params["bias"]
    else:
        bias = True
    if "weight_init" in params:
        weight_init = params["weight_init"]
    else:
        weight_init = "uniform_"
    if weight_init not in ["uniform_", "kaiming_normal_", "zeros_"]:
        raise ValueError("The received initialization method <<{}>> is not implemented. \
                          Choose from [uniform_, kaiming_normal_, zeros_]".format(
            weight_init))

    if "backwards_weight_init" in params:
        backwards_weight_init = params["backwards_weight_init"]
    else:
        backwards_weight_init = "uniform_"
    if backwards_weight_init not in ["uniform_", "kaiming_normal_", "zeros_"]:
        raise ValueError("The received initialization method <<{}>> is not implemented. \
                          Choose from [uniform_, kaiming_normal_, zeros_]".format(
            backwards_weight_init))

    weight_rescale = params["weight_rescale"] if "weight_rescale" in params else 1
    back_weight_rescale = params["backwards_weight_rescale"] if "backwards_weight_rescale" in params else 1

    if model_type in ['dyn_pseudo', 'DRL']:

        # bw learning rate can be given as an array or single value
        if not isinstance(params["backwards_learning_rate"], list):
            backwards_learning_rate = [params["backwards_learning_rate"]] * (len(layers) - 1)
        elif len(params["backwards_learning_rate"]) == len(layers) - 1:
            backwards_learning_rate = params["backwards_learning_rate"]
        else:
            raise ValueError(f"Number of given values for backwards learning rates\
                does not match number of backward matrices ({len(layers) - 1})\
                (if all entries are equal, the value can also be given as a scalar).")

        # regularizer can be given as an array or single value
        if not isinstance(params["size_of_regularizer"], list):
            regularizer_array = [params["size_of_regularizer"]] * (len(layers) - 1)
        elif len(params["size_of_regularizer"]) == len(layers) - 1:
            regularizer_array = params["size_of_regularizer"]
        else:
            raise ValueError(f"Number of given values for the regularizer\
                does not match number of backward matrices ({len(layers) - 1})\
                (if all entries are equal, the value can also be given as a scalar).")

        if not params["regularizer_decay"]:
            regularizer_fixed = True
        else:
            regularizer_fixed = False
            regularizer_decay = params["regularizer_decay"]
        if "normalize" in params:
            normalize_inputs = params["normalize"]
        else:
            normalize_inputs = False

    # all other models have no regularizer, so set to false
    else:
        regularizer_fixed = False
        
    momentum = params["momentum"]
    weight_decay = params["weight_decay"]
    if "dataset" not in params:
        dataset_type = "mnist"
    else:
        dataset_type = params["dataset"]
    if "optimizer" not in params:
        optimizer_type = "SGD"
    else:
        optimizer_type = params["optimizer"]
    if "criterion" not in params:
        loss_criterion = "MSELoss"
    else:
        loss_criterion = params["criterion"]
    if loss_criterion not in ["MSELoss", "CrossEntropyLoss"]:
        raise ValueError("The received loss criterion <<{}>> is not implemented. \
                          Choose from [MSELoss, CrossEntropyLoss]".format(
            loss_criterion))
    if "freeze_output_layer" in params:
        freeze_output_layer = params["freeze_output_layer"]
    else:
        freeze_output_layer = False

    if dataset_type in ["yinyang", "parity"]:
        dataset_size = params["dataset_size"]
    random_seed = params["random_seed"]

    if per_images is None:
        if "per_images" in params:
            per_images = params["per_images"]
        else:
            # define how often we shall print and output
            if dataset_type == "yinyang": per_images = dataset_size // 10
            elif dataset_type == "parity": per_images = dataset_size // 2
            else: per_images = 10000


    logging.info(f'Parameters loaded.')
    logging.info(f'Loss criterion: {loss_criterion}')
    logging.info(f'Optimizer: {optimizer_type}')
    logging.info(f'Dataset: {dataset_type}')
    logging.info(f'Random seed: {random_seed}')
    logging.info(f'Bias activated' if bias else 'Bias deactivated')
    logging.info(f'Weight initialization method: {weight_init}')
    logging.info(f'Weights are rescaled by {weight_rescale}') if weight_rescale != 1 else True
    if model_type != 'backprop':
        logging.info(f'Backwards weight initialization method: {backwards_weight_init}')
        logging.info(f'Backwards weights are rescaled by {back_weight_rescale}') if back_weight_rescale != 1 else True

    logging.info(f'Learning rate: {learning_rate}')
    if model_type in ['dyn_pseudo', 'DRL']:
        logging.info(f'Backwards learning rate: {backwards_learning_rate}')
        logging.info(f'Regularizer: {regularizer_array}')
        if not regularizer_fixed:
            logging.info(f'Regularizer is dynamical. Evaluating mismatch energy every time model is saved.')
            logging.info(f'Regularizer decay: {regularizer_decay}')
        else:
            logging.info(f'Regularizer is fixed. Deactivating evaluation of mismatch energy.')
        if normalize_inputs:
            logging.info(f'Normalize active: using pseudoinverse of activations instead of transpose.')
    if freeze_output_layer:
        logging.info(f'Output layer is frozen.')

    # set random seed
    torch.manual_seed(random_seed)

    # look for gpu device, use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'The training starts on device {device}.')

    # set-up the folder to save the model
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

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
    elif dataset_type == "imagenet":
        trainset = torchvision.datasets.ImageNet(params["dataset_path"],
                                                train=True,
                                                download=True,
                                                transform=transform)
    # yinyang is not officially implemented by torchvision, so we load it by hand:
    elif dataset_type == "yinyang":
        trainset = YinYangDataset(size = dataset_size, seed = random_seed)
        trainset.classes = trainset.class_names
    # implemntation of parity dataset:
    elif dataset_type == "parity":
        trainset = ParityDataset(inputs = layers[0], samples=dataset_size, seed = random_seed)
        trainset.classes = trainset.class_names

    elif dataset_type == "mnist":
        trainset = torchvision.datasets.MNIST(params["dataset_path"],
                                              train=True,
                                              download=True,
                                              transform=transform)
    else:
        raise ValueError("The received dataset <<{}>> is not implemented. \
                          Choose from ['mnist', 'cifar10', 'imagenet', 'yinyang', 'parity']".format(
            dataset_type))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)
    nb_classes = len(trainset.classes)
    logging.info('The number of classes is %i', nb_classes)

    # for gen_pseudo and dyn_pseudo, we init a second dataloader
    if model_type in ["gen_pseudo", 'dyn_pseudo', 'DRL' ]:
        if "noise" in params:
            noise = params['noise']
            logging.info(f'Adding noise N ({noise[0]},{noise[1]}) to samples for gen-pseudo.')
        else:
            noise = None
        if "covmat" in params:
            bool_covmat = params["covmat"]
        else:
            bool_covmat = False

        if model_type == 'gen_pseudo':
            if bool_covmat: logging.info(f'Using covariance matrix for gen-pseudo.')
            else: logging.info(f'Using <rr^t> for gen-pseudo.')

        # (gen pseudo needs data to calc ds-pinv of W)
        if model_type == "gen_pseudo":
            rand_sampler = torch.utils.data.RandomSampler(
                trainset,
                num_samples=params["gen_samples"],
                replacement=True)
            genpseudo_samp = torch.utils.data.DataLoader(
                trainset,
                batch_size=params["gen_samples"],
                sampler=rand_sampler)
            genpseudo_iterator = iter(genpseudo_samp)

        # (DRL needs data in shape with batch size 1)
        if model_type == "DRL":
            DRL_sampler = torch.utils.data.RandomSampler(
                trainset,
                num_samples=len(trainset),
                replacement=True)
            DRL_samp = torch.utils.data.DataLoader(
                trainset,
                batch_size=1,
                sampler=DRL_sampler)

        # (dyn pseudo needs all data to calculate mismatch energy)
        if not regularizer_fixed and model_type == "dyn_pseudo":
            data_samp = torch.utils.data.DataLoader(
                trainset,
                batch_size=len(trainset))
            if PRINT_DEBUG: timer = time.time()
            input_data = next(iter(data_samp))[0].view(len(trainset), -1)
            if PRINT_DEBUG: logging.info(f'Time to load data: {time.time()-timer}s')
            
            if dataset_type in ["yinyang", "parity"]: input_data = input_data.float()
            # calling the second dataloader changes the RNG state, so we reset
            torch.manual_seed(random_seed)
            # initialise an array to save the mismatch energies
            mm_energy = []
            # count how often mismatch energy in each layer has been calculated
            # since last update of regularizer
            mm_energy_counter = [0 for layer in range(len(layers)-1)]

    logging.info("Datasets are loaded")

    # make the networks
    net_params =   {"weight_init" : weight_init,
                    "backwards_weight_init" : backwards_weight_init,
                    "bias" : bias,
                    "weight_rescale" : weight_rescale,
                    "back_weight_rescale" : back_weight_rescale}
    backprop_net = exp_aux.load_network(model_type, 
                                        layers,
                                        net_params)
    backprop_net.to(device)

    # set up the optimizer and the loss function
    if loss_criterion == "MSELoss":
        y_onehot = torch.empty(batch_size, nb_classes, device=device)
        loss_function = torch.nn.MSELoss(reduction='sum')
    elif loss_criterion == "CrossEntropyLoss":
        loss_function = torch.nn.CrossEntropyLoss()
    if optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(
            backprop_net.parameters(), lr=learning_rate, momentum=momentum,
            weight_decay=weight_decay)
    elif optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(
            backprop_net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError("The chosen optimizer <<{}>> is not implemented. \
                          Choose from ['SGD', 'Adam']".format(
            optimizer_type))

    # optionally, freeze output layer
    if freeze_output_layer:
        backprop_net.synapses[-1].weight.requires_grad = False
        if len(layers) == 2:
            raise ValueError("All weights frozen, optimizer has nothing to do, aborting.")

    # set up scheduler for learning rate decay. KM: disabled, as no improvement seen for MNIST/BP
    #lmbda = lambda epoch: 1.0
    #scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

    # save the initial network
    file_to_save = (f"model_{model_type}_epoch_0_images_"
                    f"0.pth")
    path_to_save = os.path.join(model_folder, file_to_save)
    torch.save(backprop_net.state_dict(),
               path_to_save)

    # for dyn_pseudo, set normalizing of weights
    if model_type == "dyn_pseudo":
        for i in range(len(backprop_net.synapses)):
            backprop_net.synapses[i].normalize = normalize_inputs

    if len(trainset) % batch_size != 0:
        raise ValueError(f"Number of data vectors ({len(trainset)}) is not divisible by batch size ({batch_size}). \
                          This is required in this implementation in order to save the model every {per_images} updates.")
    # set random seed
    torch.manual_seed(random_seed)

    # train the network
    counter = 0
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        logging.info(f'• Working on epoch {epoch}')
        for index, data in enumerate(trainloader, 0):
            # redo the pseudo-inverse if applicable
            if model_type in ["pseudo_backprop", "gen_pseudo", "DRL"]:
                if counter % params["pinverse_recalc"] == 0:
                    if model_type == 'pseudo_backprop':
                        with torch.no_grad():
                            backprop_net.redo_backward_weights()
                    if model_type == 'gen_pseudo':
                        # get a subset of the dataset
                        try:
                            sub_data = genpseudo_iterator.next()[0].view(
                                params["gen_samples"], -1).to(device)
                        except StopIteration:
                            genpseudo_iterator = iter(genpseudo_samp)
                            sub_data = genpseudo_iterator.next()[0].view(
                                params["gen_samples"], -1).to(device)
                        with torch.no_grad():
                            backprop_net.redo_backward_weights(
                                dataset=sub_data.float(), noise=noise, covmat=bool_covmat)
                    if model_type == 'DRL':
                        # use trainset to learn backwards weights
                        with torch.no_grad():
                            backprop_net.redo_backward_weights(
                                dataset=DRL_samp, noise=noise, bw_lr=backwards_learning_rate, regularizer_array=regularizer_array)
                counter += 1

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # for yinyang, need to convert to float32 because data is in float64
            if dataset_type in ["yinyang", "parity"]: inputs = inputs.float()

            inputs = inputs.view(batch_size, -1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = backprop_net(inputs)
            if loss_criterion == "MSELoss":
                y_onehot.zero_()
                unsq_label = labels.unsqueeze(1)
                unsq_label.to(device)
                y_onehot.scatter_(1, unsq_label, 1)
                loss_value = loss_function(outputs, y_onehot)
            elif loss_criterion == "CrossEntropyLoss":
                loss_value = loss_function(outputs, labels)
            loss_value.backward()

            # for dyn pseudo backprop, we have a separate backwards learning rate
            # and we have to add the regularizer
            if model_type == 'dyn_pseudo':
                for i in range(len(backprop_net.synapses)):
                    # add regularizer for backwards matrix
                    backprop_net.synapses[i].weight_back.grad += regularizer_array[i] * backprop_net.synapses[i].get_backward()
                    # multiply by backwards learning rate
                    # (optimizer multiplies this with learning_rate)
                    backprop_net.synapses[i].weight_back.grad *= backwards_learning_rate[i] / learning_rate

            optimizer.step()
            #scheduler.step()

            # print statistics
            # running loss is the loss measured on the last 2000 minibatches
            running_loss += loss_value.item()

            if ((index+1) * batch_size) % per_images == 0:
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

                if PRINT_DEBUG:
                    for i in range(len(backprop_net.synapses)):
                        logging.info(f"Norm of weights in layer {i}: {torch.linalg.norm(backprop_net.synapses[i].get_forward())}")
                        logging.info(f"Norm of weight update in layer {i}: {torch.linalg.norm(backprop_net.synapses[i].weight.grad)}") 
                        #logging.info(f"Weights in layer {i}: {backprop_net.synapses[i].get_forward()}")
                        logging.info(f"Weight update in layer {i}: {backprop_net.synapses[i].weight.grad}") 
                        if model_type != 'backprop':
                            #logging.info(f"Backwards weights in layer {i}: {backprop_net.synapses[i].get_backward()}")
                            logging.info(f"Norm of backwards weights in layer {i}: {torch.linalg.norm(backprop_net.synapses[i].get_backward())}")


                if PRINT_DEBUG and model_type == 'dyn_pseudo':
                    B_array = backprop_net.get_backward_weights()
                    print('Mean of bw arrays: ', [np.mean(array) for array in B_array])

                if not regularizer_fixed and model_type == 'dyn_pseudo':
                    # if the regularizer is not fixed, calculate mismatch energy
                    # to check if it needs to be updated
                    W_array = backprop_net.get_forward_weights()
                    B_array = backprop_net.get_backward_weights()

                    # if PRINT_DEBUG: gamma_timer = time.time()
                    # Gamma_array = backprop_net.get_gamma_matrix(dataset=input_data)
                    # if PRINT_DEBUG: logging.info(f'Time to calculate Gamma matrices: {time.time()-gamma_timer}s')

                    if PRINT_DEBUG: gamma_timer = time.time()
                    Gamma2_array = backprop_net.get_gamma2_matrix(dataset=input_data)
                    if PRINT_DEBUG: logging.info(f'Time to calculate squared Gamma matrices: {time.time()-gamma_timer}s')
                    
                    if PRINT_DEBUG: timer = time.time()
                    mm_energy.append(
                            [ calc_mismatch_energy_fast(
                                Gamma2_array[i].numpy(), B_array[i].T, W_array[i], regularizer_array[i]
                                )
                              for i in range(len(backprop_net.synapses))]
                        )
                    if PRINT_DEBUG: logging.info(f'Time to calculate mismatch energy: {time.time()-timer}s')

                    mm_energy_counter = [x + 1 for x in mm_energy_counter]

                    for i in range(len(backprop_net.synapses)):
                        logging.info(f'Mismatch energy in layer {i}: {mm_energy[-1][i]}')
                        # if mm energy has been calculated enough times, check if it has reached a plateau
                        if mm_energy_counter[i] >= 5:
                            # calculate relative error of last 5 mm_energies
                            if np.sqrt(np.cov(np.array(mm_energy).T[i][-5:])) < 1e-2 * np.mean(np.array(mm_energy).T[i][-5:]):
                                logging.info(f'Plateau in mismatch energy of layer {i} detected.')
                                if regularizer_array[i] > 1e-5: 
                                    regularizer_array[i] *= regularizer_decay
                                    logging.info(f'Regularizer of layer {i} decreased. New size: {regularizer_array[i]}')
                                else:
                                    logging.info(f'Regularizer of layer {i} set to zero.')
                                    regularizer_array[i] = 0
                                # reset mismatch energy counter in the layer
                                mm_energy_counter[i] = 0


    logging.info('The training has finished after {} seconds'.format(time.time() - t0))

    # save the result
    logging.info("Saving the model")


if __name__ == '__main__':

    ARGS = exp_aux.parse_experiment_arguments()
    with open(ARGS.params, 'r+') as f:
        PARAMETERS = json.load(f)

    main(PARAMETERS, per_images=ARGS.per_images, num_workers=ARGS.num_workers)
