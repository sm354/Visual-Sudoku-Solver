from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
import os
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from IPython import display
import torch.optim as optim
import argparse

X = np.load("../sudoku_array_data/target_64k_images.npy")
labels = np.load("results/kmeans-minibatch/kmeans_mb_qt9c_labels.npy")

print(X)

X = -1*((X)/255. -1.) # for normalizing and making it black images

# print(X)

data_Y=torch.from_numpy(labels[:640000].astype(int))
data_X=torch.from_numpy(X[:640000].reshape(-1, 28*28))


print(data_Y)
print(labels)

print('data loaded')
print('data_X = ', data_X)
print('data_Y = ', data_Y)
print('data_X shape = ', data_X.shape)
print('data_Y shape = ', data_Y.shape)
print((data_Y.numpy() == 0).sum())

# for i in range(100):
#     print(data_Y[i])

plt.imshow(X[0].reshape((28,28)),cmap='gray')
plt.axis('off')
plt.savefig("./X[640k].jpg", dpi = 100)
