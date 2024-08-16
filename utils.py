import numpy as np
import torch

from torch.autograd import Variable
from scipy.interpolate import interp1d


def resize(X, length, axis=1):
    """Resize the temporal length using linear interpolation.
    X must be of shape (N,M,C) (channels last) or (N,C,M) (channels first),
    where N is the batch size, M is the temporal length, and C is the number
    of channels.
    If X is channels-last, use axis=1 (default).
    If X is channels-first, use axis=2.
    """
    length_orig = X.shape[axis]
    t_orig = np.linspace(0, 1, length_orig, endpoint=True)
    t_new = np.linspace(0, 1, length, endpoint=True)
    X = interp1d(t_orig, X, kind="linear", axis=axis, assume_sorted=True)(
        t_new
    )
    return X


def downsample_X(X, window_size):
    if X.shape[1] == window_size:
        print("No need to downsample")
        X_downsampled = X
    else:
        X_downsampled = resize(X, window_size)
    X_downsampled = X_downsampled.astype(
        "f4"
    )
    
    X_downsampled = np.transpose(X_downsampled, (0, 2, 1))
    # print("X transformed shape:", X_downsampled.shape)
    
    return X_downsampled


def separate_X_sensorwise(my_X, model_pick, my_device="cuda:0"):
    my_Xs = []
    if model_pick == "both_air":
        for i in range(4):
            start,end = i*3, (i+1)*3
            my_Xi = Variable(my_X[:,start:end,:]) # has shape [128, 3, 300]
            my_Xi = my_Xi.to(my_device, dtype=torch.float)
            my_Xs.append(my_Xi)
        my_x_atm = Variable(my_X[:,-1,:]) # has shape [128, 300]
        my_x_atm = my_x_atm.unsqueeze(1) # has shape [128, 1, 300]
        my_x_atm = my_x_atm.to(my_device, dtype=torch.float)
        my_Xs.append(my_x_atm)
    elif model_pick == "both_noair":
        for i in range(4):
            start,end = i*3, (i+1)*3
            my_Xi = Variable(my_X[:,start:end,:]) # has shape [128, 3, 300]
            my_Xi = my_Xi.to(my_device, dtype=torch.float)
            my_Xs.append(my_Xi)
    elif model_pick in ["arm_air","back_air"]:
        for i in range(2):
            start,end = i*3, (i+1)*3
            my_Xi = Variable(my_X[:,start:end,:]) # has shape [128, 3, 300]
            my_Xi = my_Xi.to(my_device, dtype=torch.float)
            my_Xs.append(my_Xi)
        my_x_atm = Variable(my_X[:,-1,:]) # has shape [128, 300]
        my_x_atm = my_x_atm.unsqueeze(1) # has shape [128, 1, 300]
        my_x_atm = my_x_atm.to(my_device, dtype=torch.float)
        my_Xs.append(my_x_atm)
    elif model_pick in ["arm_noair","back_noair"]:
        for i in range(2):
            start,end = i*3, (i+1)*3
            my_Xi = Variable(my_X[:,start:end,:]) # has shape [128, 3, 300]
            my_Xi = my_Xi.to(my_device, dtype=torch.float)
            my_Xs.append(my_Xi)
    
    return my_Xs

def reduce_array(arr, model_pick):
    if model_pick == "both_air":
        arr = arr[:,:,:]
    elif model_pick == "both_noair":
        arr = arr[:,:,:12]
    elif model_pick == "arm_air":
        arr = arr[:,:,6:]
    elif model_pick == "arm_noair":
        arr = arr[:,:,6:12]
    elif model_pick == "back_air":
        arr = arr[:,:,np.array([0,1,2,3,4,5,12])]
    elif model_pick == "back_noair":
        arr = arr[:,:,:6]
    return arr
