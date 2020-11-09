"""An experiment to train the mnist dataset."""
import logging
import os
import json
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from pseudo_backprop.experiments import exp_aux

logging.basicConfig(format='Train model -- %(levelname)s: %(message)s',
                    level=logging.DEBUG)


# pylint: disable=R0914,R0915,R0912,R1702
def main(params):
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
    if "dataset" not in params:
        dataset_type = "mnist"
    else:
        dataset_type = params["dataset"]

    # set random seed
    torch.manual_seed(params["random_seed"])

    # set-up the folder to save the model
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # set up the normalizer
    # Normalize the images to
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(0.5, 0.5)])

    # get the dataset
    logging.info("Loading the datasets")
    if dataset_type == "cifar10":
        trainset = torchvision.datasets.CIFAR10(params["dataset_path"],
                                                train=True,
                                                download=True,
                                                transform=transform)
    elif dataset_type == "mnist":
        trainset = torchvision.datasets.MNIST(params["dataset_path"],
                                              train=True,
                                              download=True,
                                              transform=transform)
    else:
        raise ValueError("The received dataset <<{}>> is not implemented. \
                          Choose from ['mnist', 'cifar10']".format(
            dataset_type))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    if model_type == "gen_pseudo":
        rand_sampler = torch.utils.data.RandomSampler(trainset,
                                                      num_samples=500000,
                                                      replacement=True)
        genpseudo_samp = torch.utils.data.DataLoader(
            trainset,
            batch_size=params["gen_samples"],
            sampler=rand_sampler)
        genpseudo_iterator = iter(genpseudo_samp)

    logging.info("Datasets are loaded")

    # make the networks
    backprop_net = exp_aux.load_network(model_type, layers)

    # set up the optimizer and the loss function
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        backprop_net.parameters(), lr=learning_rate, momentum=momentum)

    # save the initial network
    file_to_save = (f"model_{model_type}_epoch_0_images_"
                    f"0.pth")
    path_to_save = os.path.join(model_folder, file_to_save)
    torch.save(backprop_net.state_dict(),
               path_to_save)

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
                        logging.info('counter type called')
                        try:
                            sub_data = genpseudo_iterator.next()[0].view(
                                params["gen_samples"], -1)
                        except StopIteration:
                            genpseudo_iterator = iter(genpseudo_samp)
                            sub_data = genpseudo_iterator.next()[0].view(
                                params["gen_samples"], -1)
                        with torch.no_grad():
                            backprop_net.redo_backward_weights(
                                            dataset=sub_data)
                counter += 1

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
            if index % (10000/batch_size) == 999:
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

    logging.info('The training has finished')

    # save the result
    logging.info("Saving the model")


if __name__ == '__main__':

    ARGS = exp_aux.parse_experiment_arguments()
    with open(ARGS.params, 'r+') as f:
        PARAMETERS = json.load(f)

    main(PARAMETERS)
