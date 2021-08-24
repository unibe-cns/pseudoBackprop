"""Experiment to test whether dynamical pseudobackprop converges 
   to the data-specific pseudoinverse of the training data"""
import logging
import os
import json
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch import nn
from pseudo_backprop.experiments import exp_aux
from pseudo_backprop.experiments.yinyang_dataset.dataset import YinYangDataset
from pseudo_backprop.aux import evaluate_model
from pseudo_backprop.aux import loss_error
import exp_pseudo_backprop.visualization as visu
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt


torch.autograd.set_detect_anomaly(True)
logging.basicConfig(format='Evaluate model -- %(levelname)s: %(message)s',
                    level=logging.INFO)


# pylint: disable=R0914,R0915,R0912,R1702
def main(params, val_epoch = None, per_images = None, num_workers = 0):

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
        raise ValueError("""Invalid model type. This action can only\
            be run for dynamical pseudobackprop""")
    if "bias" in params:
            bias = params["bias"]
    else:
        bias = True


    if dataset_type == "yinyang":
        dataset_size = params["dataset_size"]
        random_seed = params["random_seed"]
    if "criterion" not in params:
        loss_criterion = "MSELoss"
    else:
        loss_criterion = params["criterion"]

    # set random seed
    torch.manual_seed(params["random_seed"])

    # look for gpu device, use gpu if available
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)
    nb_classes = len(trainset.classes)

    logging.info("Datasets are loaded")

    # make the networks
    net_params =   {"bias" : bias}
    backprop_net = exp_aux.load_network(model_type, 
                                        layers,
                                        net_params)
    if per_images is None:
        if "per_images" in params:
            per_images = params["per_images"]
        else:
            # define how often we shall print and output
            if dataset_type == "yinyang": per_images = dataset_size // 10
            elif dataset_type == "parity": per_images = dataset_size // 2
            else: per_images = 10000

    # make a dataloader for the training set
    genpseudo_samp = torch.utils.data.DataLoader(
            trainset,
            batch_size=len(trainset))
    genpseudo_iterator = iter(genpseudo_samp)

    sub_data = genpseudo_iterator.next()[0].view(len(trainset), -1).to(device)

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
        per_images = dataset_size // 2
        nb_batches = int(dataset_size / per_images)

    # load the saved network states and calculate cosine similarity
    loss_array = []
    conf_matrix_array = {}
    error_ratio_array = []

    fw_weights_array = []
    back_weights_array = []
    fw_norm_weights = [0]*(len(layers)-1)
    back_norm_weights = [0]*(len(layers)-1)
    fw_norm_weights_array = []
    back_norm_weights_array = []
    cos_trans_array = []
    cos_trans = [0]*(len(layers)-1)
    cos_pinv_array = []
    cos_pinv = [0]*(len(layers)-1)
    cos_dspinv_array = []
    cos_dspinv = [0]*(len(layers)-1)
    cos_vecs_trans_array = []
    cos_vecs_trans = [0]*(len(layers)-1)
    cos_vecs_pinv_array = []
    cos_vecs_pinv = [0]*(len(layers)-1)
    cos_vecs_dsp_array = []
    cos_vecs_dsp = [0]*(len(layers)-1)
    dist_array = []
    dist = [0]*(len(layers)-1)

    epoch_array = []
    image_array = []

    cos_sim_vec = nn.CosineSimilarity(dim=0, eps=1e-6)

    for index in range(epochs * nb_batches + 1):
        epoch = 0 if index == 0 else (index - 1) // nb_batches
        ims = 0 if index == 0 else (((index - 1) % nb_batches) + 1) \
            * per_images

        if val_epoch != None:
            if epoch != val_epoch: continue

        epoch_array.append(epoch)
        image_array.append(ims)

        file_to_load = (f"model_{model_type}_epoch_{epoch}_images_"
                        f"{ims}.pth")
        logging.info(f'• Processing model at state of epoch {epoch} and image {ims}.')
        path_to_model = os.path.join(model_folder, file_to_load)
        try:
            backprop_net.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
        except FileNotFoundError:
            logging.info(f'File not found. Check that model has trained successfully.')
            break

        # evaluate the model using the train data set
        loss, confusion_matrix = evaluate_model(backprop_net, trainloader,
                                                batch_size, device,
                                                nb_classes, loss_criterion)
        # load the error vector for later usage
        error_vec_out = loss_error(backprop_net, trainloader,
                                             batch_size, device,
                                             nb_classes)
        class_ratio = (confusion_matrix.diagonal().sum() /
                       confusion_matrix.sum())
        loss_array.append(loss)
        error_ratio_array.append(1 - class_ratio)

        logging.info(f'The final classification ratio is: {class_ratio}')
        # logging.info(f'The final loss function: {loss}')
        # logging.info(f'The final confusion matrix is:\n {confusion_matrix}')

        # for yinyang, need to convert to float32 because data is in float64
        if dataset_type in ["yinyang", "parity"]: sub_data = sub_data.float()

        # extract the backwards matrix at this stage
        fw_weights_array.append(backprop_net.get_forward_weights().copy())
        back_weights_array.append(backprop_net.get_backward_weights().copy())

        # generate a list of the data-specific pinverse matrices
        logging.info("Calculating data-specific pseudoinverse matrices")

        try:
            dataspecPinv_array = backprop_net.get_dataspec_pinverse(dataset=sub_data)
        except(np.linalg.LinAlgError):
            logging.info("SVD did not converge. Skipping")
            continue

        # error vectors per layer are saved in order from output to first layer
        error_vecs_B = [error_vec_out]

        # append error vectors in other layers:
        for layer in range(len(layers) -2, -1, -1):
        
            error_vecs_B.append(
                torch.matmul(
                    torch.from_numpy(back_weights_array[-1][layer].T), 
                        error_vecs_B[-1]))

        # calculate same for transpose of W to compare
        # this fictional error in layer i is given by backpropagating all errors with W^T
        # i.e. the mismatch will increase from last to first layer
        error_vecs_trans = [error_vec_out]
        for layer in range(len(layers) -2, -1, -1):
        
            error_vecs_trans.append(
                torch.matmul(
                    torch.from_numpy(fw_weights_array[-1][layer].T), 
                        error_vecs_trans[-1]))

        # calculate same for ds-pinv of W to compare
        error_vecs_dspinv = [error_vec_out]
        for layer in range(len(layers) -2, -1, -1):
        
            error_vecs_dspinv.append(
                torch.matmul(
                    dataspecPinv_array[layer].float(), 
                        error_vecs_dspinv[-1]))

        # calculate same for pinv of W to compare
        error_vecs_pinv = [error_vec_out]
        for layer in range(len(layers) -2, -1, -1):
        
            error_vecs_pinv.append(
                torch.matmul(
                    torch.linalg.pinv(torch.from_numpy(fw_weights_array[-1][layer])), 
                        error_vecs_pinv[-1]))

        # note: error vectors are order from output to first layer
        # print(error_vecs)

        for layer in range(len(layers)-1):
            logging.info(f'# Layer {layer}')

            # calculate the cosine similarity using the Frobenius norm
            # between the error backpropagated using the tranpose of the weights
            # and the dynamical backwards matrix
            cos_vecs_trans[layer] = np.round(
                cos_sim_vec(error_vecs_B[-1-layer], error_vecs_trans[-1-layer]).tolist()
                ,6)
                
            if cos_vecs_trans[layer] > 1 or cos_vecs_trans[layer] < -1:
                raise ValueError(f"Cosine between tensors has returned invalid value {cos_vecs_trans[layer]}")
            logging.info(f'The cosine between the errors propagated using the '
                         f'backwards weights and the transpose of the forward weights '
                         f'in layer {layer} is: {cos_vecs_trans[layer]}')

            # calculate the cosine similarity using the Frobenius norm
            # between the tranpose of the weights
            # and the dynamical backwards matrix
            cos_trans[layer] = np.round(
                exp_aux.cosine_similarity_tensors(
                    torch.from_numpy(back_weights_array[-1][layer].T),
                    torch.from_numpy(fw_weights_array[-1][layer].T)
                    ).tolist()
                ,6)
            if cos_trans[layer] > 1 or cos_trans[layer] < -1:
                raise ValueError(f"Cosine between tensors has returned invalid value {cos_trans[layer]}")
            logging.info(f'The cosine between the backwards weights and the transpose of the forward weights '
                                 f'in layer {layer} is: {cos_trans[layer]}')

            # calculate the cosine similarity using the Frobenius norm
            # between the error backpropagated using the pseudoinverse
            # and the dynamical backwards matrix
            cos_vecs_pinv[layer] = np.round(
                cos_sim_vec(error_vecs_B[-1-layer], error_vecs_pinv[-1-layer]).tolist()
                ,6)
                
            if cos_vecs_pinv[layer] > 1 or cos_vecs_pinv[layer] < -1:
                raise ValueError(f"Cosine between tensors has returned invalid value {cos_vecs_pinv[layer]}")
            logging.info(f'The cosine between the errors propagated using the '
                         f'backwards weights and the pseudoinverse '
                         f'in layer {layer} is: {cos_vecs_pinv[layer]}')

            # calculate the cosine similarity using the Frobenius norm
            # between the pseudoinverse
            # and the dynamical backwards matrix
            cos_pinv[layer] = np.round(
                exp_aux.cosine_similarity_tensors(
                    torch.from_numpy(back_weights_array[-1][layer].T),
                    torch.linalg.pinv(torch.from_numpy(fw_weights_array[-1][layer]))
                    ).tolist()
                ,6)
            if cos_pinv[layer] > 1 or cos_pinv[layer] < -1:
                raise ValueError(f"Cosine between tensors has returned invalid value {cos_pinv[layer]}")
            logging.info(f'The cosine between the backwards weights and the pseudoinverse '
                                 f'in layer {layer} is: {cos_pinv[layer]}')

            # calculate the cosine similarity using the Frobenius norm
            # between the error backpropagated using the data-specific pseudoinverse
            # and the dynamical backwards matrix
            cos_vecs_dsp[layer] = np.round(
                cos_sim_vec(error_vecs_B[-1-layer], error_vecs_dspinv[-1-layer]).tolist()
                ,6)
                
            if cos_vecs_dsp[layer] > 1 or cos_vecs_dsp[layer] < -1:
                raise ValueError(f"Cosine between tensors has returned invalid value {cos_vecs_dsp[layer]}")
            logging.info(f'The cosine between the errors propagated using the '
                         f'backwards weights and the data-specific pseudoinverse '
                         f'in layer {layer} is: {cos_vecs_dsp[layer]}')

            # calculate the cosine similarity using the Frobenius norm
            # between the data-specific pseudoinverse
            # and the dynamical backwards matrix
            cos_dspinv[layer] = np.round(
                exp_aux.cosine_similarity_tensors(
                    torch.from_numpy(back_weights_array[-1][layer].T),
                    dataspecPinv_array[layer].float()
                    ).tolist()
                ,6)
            if cos_dspinv[layer] > 1 or cos_dspinv[layer] < -1:
                raise ValueError(f"Cosine between tensors has returned invalid value {cos_dspinv[layer]}")
            logging.info(f'The cosine between the backwards weights and the data-specific pseudoinverse '
                                 f'in layer {layer} is: {cos_dspinv[layer]}')



            # as an alternative measure of convergence, we also
            # calculate the distance between tensors
            # dist[layer] = np.round(
            #     exp_aux.norm_distance(
            #         torch.from_numpy(back_weights_array[-1][layer].T),
            #         torch.linalg.pinv(torch.from_numpy(fw_weights_array[-1][layer]))
            #         ).tolist()
            #     ,6)
            # logging.info(f'The distance between the backwards weights and the (!) pseudoinverse '
            #                      f'in layer {layer} is: {dist[layer]}')

            # dist[layer] = np.round(
            #     exp_aux.norm_distance(
            #         torch.from_numpy(back_weights_array[-1][layer].T),
            #         dataspecPinv_array[layer].float()
            #         ).tolist()
            #     ,6)
            # logging.info(f'The distance between the backwards weights and the data-specific pseudoinverse '
            #                      f'in layer {layer} is: {dist[layer]}')

            # calculate norm of weights for later analysis
            # fw_norm_weights[layer] = torch.linalg.norm(torch.from_numpy(fw_weights_array[-1][layer]))
            # logging.info(f'The norm of the forward weights in layer {layer} is: {fw_norm_weights[layer]}')
            # back_norm_weights[layer] = torch.linalg.norm(torch.from_numpy(back_weights_array[-1][layer]))
            # logging.info(f'The norm of the backward weights in layer {layer} is: {back_norm_weights[layer]}')
        
        # dist_array.append(dist.copy())
        cos_trans_array.append(cos_trans.copy())
        cos_pinv_array.append(cos_pinv.copy())
        cos_dspinv_array.append(cos_dspinv.copy())

        cos_vecs_trans_array.append(cos_vecs_trans.copy())
        cos_vecs_dsp_array.append(cos_vecs_dsp.copy())
        cos_vecs_pinv_array.append(cos_vecs_pinv.copy())

        fw_norm_weights_array.append(fw_norm_weights.copy())
        back_norm_weights_array.append(back_norm_weights.copy())

    # plot results
    fig_vecs, axes_vecs = plt.subplots(len(layers), 1,figsize=(8,3*(len(layers)+1)))
    fig_mats, axes_mats = plt.subplots(len(layers), 1,figsize=(8,3*(len(layers)+1)))

    visu.evaluation(axes_vecs,
                      epoch_array=epoch_array,
                      data_arrays= [np.array(cos_vecs_trans_array).T,
                                    np.array(cos_vecs_pinv_array).T,
                                    np.array(cos_vecs_dsp_array).T],
                      error_array=100*np.array(error_ratio_array),
                      limits=[-1,1],
                      labels=  ["cos of errors B vs. W^T",
                                "cos of errors B vs. pinv(W)",
                                "cos of errors B vs. ds-pinv(W)"],
                      linestyles=['-.', '--', '-']
                    )

    visu.evaluation(axes_mats,
                      epoch_array=epoch_array,
                      data_arrays= [np.array(cos_trans_array).T,
                                    np.array(cos_pinv_array).T,
                                    np.array(cos_dspinv_array).T],
                      error_array=100*np.array(error_ratio_array),
                      limits=[-1,1],
                      labels=  ["cos of B vs. W^T",
                                "cos of B vs. pinv(W)",
                                "cos of B vs. ds-pinv(W)"],
                      linestyles=['-.', '--', '-']
                    )

    # # Add labels
    # axes[(0, 0)].text(-0.25, 0.30, "Training set", rotation=90,
    #                   transform=axes[(0, 0)].transAxes, ha="center",
    #                   weight='bold')
    # axes[(1, 0)].text(-0.25, 0.40, "Test set", rotation=90,
    #                   transform=axes[(1, 0)].transAxes, ha="center",
    #                   weight="bold")

    # network_string = "["
    # for layer in params["layers"]:
    #     network_string += str(layer) + ", "
    # network_string = network_string[:-2] + "]"

    # suptext = f"{param_data['dataset']} data set, network: {network_string}"
    # fig.suptitle(suptext)

    logging.info(f'Results plotted and saved as eval_error_vecs.png and eval_matrices.png')
    fig_vecs.savefig('eval_error_vecs.png')
    fig_mats.savefig('eval_matrices.png')

    # Save the results into an appropriate file into the model folder
    layer_names = [str(i) for i in list(range(len(cos_trans_array[-1])))]
    to_save = np.array([epoch_array, image_array, error_ratio_array])
    to_save = np.concatenate(
        (to_save, np.array(fw_norm_weights_array).T,
            np.array(back_norm_weights_array).T,
            np.array(cos_vecs_trans_array).T,
            np.array(cos_vecs_dsp_array).T,
            np.array(cos_vecs_pinv_array).T,
            np.array(cos_trans_array).T,
            np.array(cos_pinv_array).T,
            np.array(cos_dspinv_array).T),
        axis=0).T
    file_to_save_cos = os.path.join(model_folder, f'train_results_dyn_pseudo.csv')
    header = ('epochs, images, error, '
             + 'W layer ' + ', W layer '.join([layer for layer in layer_names])
             + ', B layer ' + ', B layer '.join([layer for layer in layer_names])
             + ', cos vec W.T layer ' + ', cos vec W.T layer '.join([layer for layer in layer_names])
             + ', cos vec pinv layer ' + ', cos vec pinv layer '.join([layer for layer in layer_names])
             + ', cos vec dspinv layer ' + ', cos vec dspinv layer '.join([layer for layer in layer_names])
             + ', cos mat W.T layer ' + ', cos mat W.T layer '.join([layer for layer in layer_names])
             + ', cos mat pinv layer ' + ', cos mat pinv layer '.join([layer for layer in layer_names])
             + ', cos mat dspinv layer ' + ', cos mat dspinv layer '.join([layer for layer in layer_names])
             # + ', dist mat layer ' + ', dist mat layer '.join([layer for layer in layer_names])
             )
    np.savetxt(file_to_save_cos, to_save, delimiter=',',
                       header=header)
    logging.info(f'Saved results to train_results_dyn_pseudo.csv')



if __name__ == '__main__':

    ARGS = exp_aux.parse_experiment_arguments()
    with open(ARGS.params, 'r+') as f:
        PARAMETERS = json.load(f)
    if ARGS.epoch != None:
        EPOCH = ARGS.epoch
    else:
        EPOCH = None

    main(PARAMETERS, EPOCH, per_images=ARGS.per_images, num_workers=ARGS.num_workers)
