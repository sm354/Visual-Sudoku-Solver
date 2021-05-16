import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
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
dataloader = DataLoader(X, batch_size=1000, shuffle=False)

oneshot_data = oneshot_data.reshape((-1, 1, 28, 28))/(255.)
oneshot_data = np.concatenate((oneshot_data, oneshot_data, oneshot_data), axis = 1)
oneshot_data = torch.from_numpy(oneshot_data)

print('shape of X = ', X.shape)
print('shape of x_oneshot = ', oneshot_data.shape)

print('X = \n', X)
print('x_oneshot = \n', oneshot_data)

vgg16.eval()
X_vgg = []
for i, (x) in enumerate(dataloader):
    # print('x = ', x)
    print('i = ', i)
    # x = upsampler(x)
    # print(x.shape)
    x = vgg16(x.float())
    X_vgg.append(x)
    # if(i<5):
    #     break
X_vgg = torch.cat(X_vgg)

oneshot_data = upsampler(oneshot_data)
oneshot_data_vgg = vgg16(oneshot_data.float())

print('X_vgg shape = ', X_vgg.shape)
print('oneshot_data_vgg shape', oneshot_data_vgg.shape)

#save vgg embeddings X_vgg and oneshot_data_cgg
X_vgg = X_vgg.detach().numpy()
oneshot_data_vgg = oneshot_data_vgg.detach().numpy()

np.save("/home/ee/btech/ee1180957/scratch/Harman/DL-ASS2/COL870-Assignment-2/results/vggnet_embeddings/X_query_target_vggnet", X_vgg)
np.save("/home/ee/btech/ee1180957/scratch/Harman/DL-ASS2/COL870-Assignment-2/results/vggnet_embeddings/oneshot_data_vggnet", oneshot_data)



