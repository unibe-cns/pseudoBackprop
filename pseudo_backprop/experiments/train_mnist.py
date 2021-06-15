"""An experiment to train the mnist dataset."""
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
from pseudo_backprop.aux import calc_mismatch_energy

torch.autograd.set_detect_anomaly(True)
logging.basicConfig(format='Train model -- %(levelname)s: %(message)s',
                    level=logging.DEBUG)

PRINT_DEBUG = True

# pylint: disable=R0914,R0915,R0912,R1702
def main(params):

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
    if model_type == 'dyn_pseudo':
        backwards_learning_rate = params["backwards_learning_rate"]
        size_of_regularizer = params["size_of_regularizer"]
        regularizer_decay = params["regularizer_decay"]
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

    if dataset_type == "yinyang":
        dataset_size = params["dataset_size"]
    random_seed = params["random_seed"]

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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    nb_classes = len(trainset.classes)
    logging.info('The number of classes is %i', nb_classes)

    # for gen_pseudo and dyn_pseudo, we init a second dataloader
    if model_type == "gen_pseudo" or 'dyn_pseudo':
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
        # (dyn pseudo needs all data to calculate mismatch energy)
        if model_type == "dyn_pseudo":
            data_samp = torch.utils.data.DataLoader(
                trainset,
                batch_size=len(trainset))

    logging.info("Datasets are loaded")

    # make the networks
    backprop_net = exp_aux.load_network(model_type, layers)
    backprop_net.to(device)

    for i in range(len(backprop_net.synapses)):
        print(backprop_net.synapses[i].weight_back)

    # set up the optimizer and the loss function
    y_onehot = torch.empty(batch_size, nb_classes, device=device)
    loss_function = torch.nn.MSELoss(reduction='sum')
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

    # set up scheduler for learning rate decay. KM: disabled, as no improvement seen for MNIST/BP
    #lmbda = lambda epoch: 1.0
    #scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

    # save the initial network
    file_to_save = (f"model_{model_type}_epoch_0_images_"
                    f"0.pth")
    path_to_save = os.path.join(model_folder, file_to_save)
    torch.save(backprop_net.state_dict(),
               path_to_save)

    # define how often we shall print and output
    if dataset_type == "yinyang": per_images = dataset_size // 10
    else: per_images = 10000

    # for dyn pseudo, calculate the matrix Gamma
    # (sqrt of the autocorrelation) to calculate mismatch energy
    if model_type == 'dyn_pseudo':
        sub_data = next(iter(data_samp))[0].view(len(trainset), -1)
        Gamma_array = backprop_net.get_gamma_matrix(dataset=sub_data)
        # initialise an array to save the mismatch energies
        mm_energy = []
        # print(torch.linalg.norm(Gamma_array[1]))
        # count how often mismatch energy in each layer has been calculated
        # since last update of regularizer
        mm_energy_counter = [0 for layer in range(len(layers)-1)]
        regularizer_array = [size_of_regularizer for layer in range(len(layers)-1)]

        # if PRINT_DEBUG:
        #     # print mismatch energy at minimum, i.e. bachwards weights = ds-pinv
        #     dspinv = backprop_net.get_dataspec_pinverse(sub_data)
        #     W_array = backprop_net.get_forward_weights()
        #     mm_energy_minimum = [ calc_mismatch_energy(
        #                         Gamma_array[i].numpy(), dspinv[i].float().numpy(), W_array[i], 0.*regularizer_array[i]
        #                         )
        #                       for i in range(len(backprop_net.synapses))]
        #     logging.info(f'Minimum of mismatch energy (using data-specific pseudoinverse): {mm_energy_minimum}')

        # # FOR TESTING, set bw weights to ds-pinv
        # logging.info(f'Calculating ds-pinvs')
        # dspinv = backprop_net.get_dataspec_pinverse(sub_data)
        # for i in range(len(backprop_net.synapses)):
        #     backprop_net.synapses[i].weight_back = torch.nn.Parameter(dspinv[i].t().float())

    # train the network
    counter = 0
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        logging.info(f'• Working on epoch {epoch}')
        for index, data in enumerate(tqdm(trainloader), 0):
            # redo the pseudo-inverse if applicable
            if model_type in ["pseudo_backprop", "gen_pseudo"]:
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
                            genpseudo_iterator = iter(data_samp)
                            sub_data = genpseudo_iterator.next()[0].view(
                                params["gen_samples"], -1).to(device)
                        with torch.no_grad():
                            backprop_net.redo_backward_weights(
                                dataset=sub_data)
                counter += 1

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # for yinyang, need to convert to float32 because data is in float64
            if dataset_type == "yinyang": inputs = inputs.float()

            inputs = inputs.view(batch_size, -1)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            y_onehot.zero_()
            unsq_label = labels.unsqueeze(1)
            unsq_label.to(device)
            y_onehot.scatter_(1, unsq_label, 1)
            outputs = backprop_net(inputs)
            loss_value = loss_function(outputs, y_onehot)
            loss_value.backward()

            # for dyn pseudo backprop, we have a separate backwards learning rate
            # and we have to add the regularizer
            if model_type == 'dyn_pseudo':
                for i in range(len(backprop_net.synapses)):
                    # add regularizer for backwards matrix
                    # print('Frobenius norm of update of backwards weights for synapse BEFORE regularizer', i,
                    # ':', torch.linalg.norm(backprop_net.synapses[i].weight_back.grad))

                    backprop_net.synapses[i].weight_back.grad += regularizer_array[i] * backprop_net.synapses[i].get_backward()

                    # print('Frobenius norm of update of backwards weights for synapse AFTER regularizer', i,
                    # ':', torch.linalg.norm(backprop_net.synapses[i].weight_back.grad))
                    # the optimizer applies the standard learning rate on all parameter updates
                    # So in order to implement a custom learning rate for the backwards matrix,
                    # we rescale the gradient of the backwards weights here
                    # print('before learning_rate:', torch.linalg.norm(backprop_net.synapses[i].weight_back.grad))
                    backprop_net.synapses[i].weight_back.grad *= backwards_learning_rate / learning_rate
                    # print('after learning_rate:', torch.linalg.norm(backprop_net.synapses[i].weight_back.grad))

            

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

                if model_type == 'dyn_pseudo':
                    # calculate mismatch energy
                    W_array = backprop_net.get_forward_weights()
                    B_array = backprop_net.get_backward_weights()
                    
                    mm_energy.append(
                            [ calc_mismatch_energy(
                                Gamma_array[i].numpy(), B_array[i].T, W_array[i], regularizer_array[i]
                                )
                              for i in range(len(backprop_net.synapses))]
                        )
                    mm_energy_counter = [x + 1 for x in mm_energy_counter]
                    # print(mm_energy)

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

                if PRINT_DEBUG and model_type == 'dyn_pseudo':
                # print the grad of the forward weights
                    # for i in range(len(backprop_net.synapses)):
                    #     print('Frobenius norm of forward weights for synapse:', i)
                    #     print(torch.linalg.norm(backprop_net.synapses[i].get_forward()))
                    # print the grad of the backwards weights
                    for i in range(len(backprop_net.synapses)):
                        # print('Grad of backwards weights for synapse:', i)
                        # print(backprop_net.synapses[i].weight_back.grad)
                        print('Frobenius norm of update of backwards weights for synapse', i,
                        ':', torch.linalg.norm(backprop_net.synapses[i].weight_back.grad))

                # for i in range(len(backprop_net.synapses)):
                #     print('Backwards weights for synapse:', i)
                #     print(backprop_net.synapses[i].get_backward())
                #     print('Grad of backwards weights for synapse:', i)
                #     print(backprop_net.synapses[i].weight_back.grad)
                     #    print('Frobenius norm of backwards weights ', i,
                     #     ':', torch.linalg.norm(backprop_net.synapses[i].get_backward()))
                     #    print('Frobenius norm of update of backwards weights for synapse for synapse ', i,
                     # ':', torch.linalg.norm(backprop_net.synapses[i].weight_back.grad))


    logging.info('The training has finished after {} seconds'.format(time.time() - t0))

    # save the result
    logging.info("Saving the model")


if __name__ == '__main__':

    ARGS = exp_aux.parse_experiment_arguments()
    with open(ARGS.params, 'r+') as f:
        PARAMETERS = json.load(f)

    main(PARAMETERS)
