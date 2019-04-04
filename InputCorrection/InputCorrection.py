# This code is mostly from "Texture Synthesis Using Convolutional Neural Networks" (Gatys et al., NIPS 2015)
# Please cite this paper if you use it.
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

def get_indices(net, constraints):
    indices = [ndx for ndx,layer in enumerate(net.blobs.keys()) if layer in constraints.keys()]
    return net.blobs.keys(),indices[::-1]

def InputCorrection(reg_param, net, init_map, layer_name, constraints):
     #get indices for gradient
    layers, indices = get_indices(net, constraints)

    global pred_result 
    global num_iter
    num_iter = 0

    #function to minimize
    def f(x):
        global num_iter
        global pred_result

        x = x.reshape(*net.blobs[layer_name].data.shape)
        net.blobs[layer_name].data[...] = x
        net.forward()
        iact_map = net.blobs[layer_name].data.copy()
        f_val = 0
        num_iter += 1
        #clear gradient in all layers
        for index in indices:
            net.blobs[layers[index]].diff[...] = np.zeros_like(net.blobs[layers[index]].diff)


        constraints['sig_pred'].parameter_lists[0].update({'activations': net.blobs['sig_pred'].data.copy()})
        
        val, grad, f_max = constraints['sig_pred'].loss_functions[0](**constraints['sig_pred'].parameter_lists[0])
        pred_result = net.blobs['sig_pred'].data[:, :, :, :].copy()
        if f_max < 0.5:
            return [val, np.array(np.zeros(np.shape(iact_map)).ravel(), dtype=float)]

        f_val = val.copy()
        f_val += reg_param*(np.sum(np.power(iact_map-init_map, 2.0)))

        net.blobs['sig_pred'].diff[...] = grad.copy()

        f_grad = net.backward(start='sig_pred')[layer_name].copy()   # last layer
        f_grad += 2*reg_param*(iact_map-init_map)
        return [f_val, np.array(f_grad.ravel(), dtype=float)]
        
    result = fmin_l_bfgs_b(func=f, x0=init_map.copy(), maxiter=100, m=20, factr=0, pgtol=1e-9)

    return result, pred_result


