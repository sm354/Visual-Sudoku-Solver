import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np
vgg16 = models.vgg16(pretrained=True)

print(vgg16)
upsampler = nn.Upsample(size = (224, 224), mode = 'bilinear', align_corners=True)
# def load_data(datapath, oneshot_data_path, nclusters):

print("loading data--------------------")

X = np.load('/home/ee/btech/ee1180957/scratch/Harman/DL-ASS2/sudoku_array_data/target_64k_images.npy')
oneshot_data = np.load('/home/ee/btech/ee1180957/scratch/Harman/DL-ASS2/Assignment 2/sample_images.npy')

X = (X)/255. #normalization
X = X.reshape((-1, 1, 28, 28))
X = np.concatenate((X, X, X), axis = 1) #(b, 3, 28, 28)
X = torch.from_numpy(X)
X = upsampler(X)

oneshot_data = oneshot_data.reshape((-1, 1, 28, 28))/(255.)
oneshot_data = np.concatenate((oneshot_data, oneshot_data, oneshot_data), axis = 1)
oneshot_data = torch.from_numpy(oneshot_data)
oneshot_data = upsampler(oneshot_data)

print('shape of X = ', X.shape)
print('shape of x_oneshot = ', oneshot_data.shape)

print('X = \n', X)
print('x_oneshot = \n', oneshot_data)

vgg16.eval()
X_vgg = vgg16(X.float())
oneshot_data_vgg = vgg16(oneshot_data.float())

print('X_vgg shape = ', X_vgg.shape)
print('oneshot_data_vgg shape', oneshot_data_vgg.shape)

