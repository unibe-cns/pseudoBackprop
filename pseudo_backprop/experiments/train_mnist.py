"""An experiment to train the mnist dataset."""
import logging
import os
import json
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from pseudo_backprop.experiments import exp_aux
from pseudo_backprop.experiments.yinyang_dataset.dataset import YinYangDataset
import time


logging.basicConfig(format='Train model -- %(levelname)s: %(message)s',
                    level=logging.DEBUG)


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
    torch.manual_seed(params["random_seed"])

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

    logging.info("Datasets are loaded")

    # make the networks
    backprop_net = exp_aux.load_network(model_type, layers)
    backprop_net.to(device)

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

    # train the network
    counter = 0
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        logging.info(f'Working on epoch {epoch + 1}')
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
                            genpseudo_iterator = iter(genpseudo_samp)
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

    logging.info('The training has finished after {} seconds'.format(time.time() - t0))

    # save the result
    logging.info("Saving the model")


if __name__ == '__main__':

    ARGS = exp_aux.parse_experiment_arguments()
    with open(ARGS.params, 'r+') as f:
        PARAMETERS = json.load(f)

    main(PARAMETERS)
