# This code is mostly from "Texture Synthesis Using Convolutional Neural Networks" (Gatys et al., NIPS 2015)
# Please cite this paper if you use it.
import numpy as np

def mean_loss(activations, target_matrix, valid_matrix):
    M = np.sum(valid_matrix)

    valid_activations = np.multiply(activations, valid_matrix)
    valid_target = np.multiply(target_matrix, valid_matrix)
    f_val = np.sum(np.power((valid_activations - valid_target), 2.0))
    f_max = np.max(np.abs(valid_activations - valid_target))
    f_grad = 2*(valid_activations - valid_target)

    return f_val, f_grad, f_max

