import numpy as np
import os
import matplotlib.pyplot as plt
import PIL.Image as Image
import torch
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn import decomposition
from scipy.sparse import csr_matrix
import torchvision
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F

import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import argparse

from classifier_arabicmnist import *
def is_set_correct(array):
    # print(array)
    # print(set(array))
    if len(set(array)) >= 8:
        # print("yes")
        # print(set(array))
        return True
    # print("no")
    return False


def clustering_accuracy(labels):
    #labels are of shape (totalsmall images in all sudoku which is divisible by 64,)
    # labels = labels.reshape((labels.shape[0]//64, -1))
    # labels = labels.reshape((-1, 8, 8))
    # print(labels.shape)
    # print(labels[0])
    # print(labels[10000])

    subatomic_correct = 0

    correct = 0
    total = 0
    #now we have labels of correct shape
    final_bool_arr = np.array([True for i in range(labels.shape[0])])
    for i in range(8):
        k = i * 2 if i<4 else (i-4) * 2
        j= (i // 4) * 4
        # print(k, j)
        # if(np.all(np.apply_along_axis(is_set_correct, axis = 1, arr = labels[:, :, i])) == True or np.all(np.apply_along_axis(is_set_correct, axis = 1, arr = labels[:, i, :])) == True or np.all(np.apply_along_axis(is_set_correct, axis = 1, arr = labels[:, k:k+2, j:j+4].reshape(-1, 8))) !=True ):
        #   correct+=1
        # total+=1

        arr1 = np.apply_along_axis(is_set_correct, axis = 1, arr = labels[:, :, i])
        arr2 = np.apply_along_axis(is_set_correct, axis = 1, arr = labels[:, i, :])
        arr3 = np.apply_along_axis(is_set_correct, axis = 1, arr = labels[:, k:k+2, j:j+4].reshape(-1, 8))
        arr = arr1*arr2*arr3
        # arr = arr1*arr2
        assert(arr.shape[0] == labels.shape[0] and len(arr.shape) == 1)
        final_bool_arr *= arr
        subatomic_correct += arr1.sum() + arr2.sum() + arr3.sum()
        # subatomic_correct += arr1.sum() + arr2.sum()

    return final_bool_arr.sum()/final_bool_arr.shape[0], subatomic_correct/(3*8*labels.shape[0])

#take input classifier and target data and see what is it outputtign and also see number of constraints satisfied
device='cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

model = LeNet(9).to(device)
model.load_state_dict(torch.load("/home/ee/btech/ee1180957/scratch/Harman/DL-ASS2/COL870-Assignment-2/Joint_training/extra_models/E5_linearlayeron7_udaclassifier_classifier.pth"))
# model = torch.load("/home/ee/btech/ee1180957/scratch/Harman/DL-ASS2/COL870-Assignment-2/Joint_training/E9_kmeans_train_query_9class/classifier_trained.pth")
model.eval()

#load target dataset data
Xtarget = np.load("/home/ee/btech/ee1180957/scratch/Harman/DL-ASS2/sudoku_array_data/target_64k_images.npy")
Xtarget = -1.*(Xtarget/255. -1.)
Xtarget=torch.from_numpy(Xtarget.reshape(-1, 1, 28, 28))
batchsize = 128
target_loader = DataLoader(Xtarget, batch_size=batchsize, shuffle=False)

target_labels = predict(model, device, target_loader, True)
print('target_labels shape = ', target_labels.shape)

# target_labels = target_labels[1*64:2*64]
print(target_labels[0:64].reshape(8, 8))
print('clustering_performance = ', clustering_accuracy(target_labels.reshape(-1, 8, 8)))

print(target_labels)

torch.save(model, "/home/ee/btech/ee1180957/scratch/Harman/DL-ASS2/COL870-Assignment-2/Joint_training/extra_models/E5_linearlayeron7_udaclassifier_classifier_modelnotstate.pth")
