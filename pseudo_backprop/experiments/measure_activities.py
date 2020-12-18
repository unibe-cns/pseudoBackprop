""" Experiment to measure the activites """
import torch
import torchvision
import torchvision.transforms as transforms
from pseudo_backprop.experiments import exp_aux
import pseudo_backprop.aux as aux


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
