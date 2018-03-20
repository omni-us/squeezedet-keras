# Project: squeezeDetOnKeras
# Filename: modelLoading
# Author: Christopher Ehmann
# Date: 21.12.17
# Organisation: searchInk
# Email: christopher@searchink.com


import h5py
import numpy as np

def load_only_possible_weights(model, weights_file, verbose = False):
    """
    Sets the weights of a model manually by layer name, even if the length of each dimensions does not match.
    :param model: a keras model
    :param weights_file: a keras ckpt file
    """

    #load model file
    f = h5py.File(weights_file, 'r')


    #get list of all datasets that are kernels and biases
    kernels_and_biases_list = []
    def append_name(name):
        if "kernel" in name or "bias" in name:
            kernels_and_biases_list.append(name)
    f.visit(append_name)


    #iterate layers
    for l in model.layers:

        #get the current models weights
        w_and_b = l.get_weights()

        #check for layers that have weights and biases
        if len(w_and_b) == 2:

            #look for corresponding indices
            for kb in kernels_and_biases_list:
                if l.name + "/kernel" in kb:

                    if verbose:
                        print("Loaded weights for {}".format(l.name))

                    #get shape of both weight tensors
                    model_shape = np.array(w_and_b[0].shape)
                    file_shape =  np.array(f[kb][()].shape)

                    #check for number of axis
                    assert len(model_shape) == len(file_shape)

                    #get the minimum length of each axis
                    min_dims = np.minimum(model_shape, file_shape)

                    #build minimum indices
                    min_idx = [slice(0, x) for x in min_dims]

                    #set to weights of loaded file
                    w_and_b[0][min_idx] = f[kb][()][min_idx]


                if l.name +"/bias" in kb:
                    if verbose:
                        print("Loaded biases for {}".format(l.name))

                    #get shape of both bias tensors
                    model_shape = np.array(w_and_b[1].shape)
                    file_shape = np.array(f[kb][()].shape)

                    # check for number of axis
                    assert len(model_shape) == len(file_shape)

                    # get the minimum length of each axis
                    min_dims = np.minimum(model_shape, file_shape)

                    #build minimum indices
                    min_idx = [ slice(0,x) for x in min_dims]

                    # set to weights of loaded file
                    w_and_b[1][min_idx] = f[kb][()][min_idx]



            #update weights
            l.set_weights(w_and_b)

