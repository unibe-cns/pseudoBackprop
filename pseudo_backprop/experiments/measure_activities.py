""" Experiment to measure the activites """
import json
import torch
import torchvision
import torchvision.transforms as transforms
from pseudo_backprop.experiments import exp_aux
import pseudo_backprop.aux as aux
from pseudo_backprop import visualization as visu


def measure_activities(params, model_path, num_examples, dataset) -> list:
    """load a model from path and make experiment to get measuer the activities

    Args:
        params (dict): Description
        model_path (str): Description
        num_examples (TYPE): Description
        dataset (str): train or test

    Returns:
        list: list of activities
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # make the networks
    network = exp_aux.load_network(params['model_type'], params['layers'])
    network.load_state_dict(torch.load(model_path))

    # Load the dataset
    # Load the model and the data
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(params["dataset_path"],
                                          train=(dataset == 'train'),
                                          download=True,
                                          transform=transform)
    rand_sampler = torch.utils.data.RandomSampler(
        trainset,
        num_samples=num_examples,
        replacement=True)
    genpseudo_samp = torch.utils.data.DataLoader(
        trainset,
        batch_size=num_examples,
        sampler=rand_sampler)
    genpseudo_iterator = iter(genpseudo_samp)

    # calculate for one example
    try:
        sub_data = genpseudo_iterator.next()[0].view(
            num_examples, -1).to(device)
    except StopIteration:
        genpseudo_iterator = iter(genpseudo_samp)
        sub_data = genpseudo_iterator.next()[0].view(
            num_examples, -1).to(device)

    act_arr = aux.calc_activities(network, sub_data,
                                  len(params['layers']))

    return act_arr


def main(params, args):
    """The experiment to be called for the command line

    Args:
        params (dict): Description
        args (args object): Description

    Raises:
        NotImplementedError: only implemented for the mnist dataset
    """

    # this is only implemented for MNIST --> check for it
    if params["dataset"] != "mnist":
        raise NotImplementedError("This experiment is only implemnented for"
                                  "the mnist dataset!")
    gen_samples = 2000

    # make the initial plot
    path_stump = f'{params["model_folder"]}/model_{params["model_type"]}'
    path_to_model = f'{path_stump}_epoch_0_images_0.pth'
    act_arr = measure_activities(params, path_to_model, gen_samples,
                                 args.dataset)
    fig = visu.plot_activities(act_arr)
    fig.savefig(f'{params["model_folder"]}/activities_final.png')

    # make the plots for the epochs as well
    for index in range(params['epochs']):
        path_to_model = f'{path_stump}_epoch_{index}_images_60000.pth'
        act_arr = measure_activities(params, path_to_model, gen_samples,
                                     args.dataset)
        fig = visu.plot_activities(act_arr)
        fig.savefig(f'{params["model_folder"]}/activities_epoch_{index+1}.png')


if __name__ == '__main__':
    arguments = exp_aux.parse_experiment_arguments()
    with open(arguments.params, 'r+') as infile:
        parameters = json.load(infile)
    main(parameters, arguments)
