import os
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import PIL.Image as Image
import torch
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn import decomposition
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--train_datapath', type=str, default = None)
parser.add_argument('--target_array_file', type=str, default = None)
parser.add_argument('--query_array_file', type=str, default = None)
parser.add_argument('--query_target_array_file', type=str, default = None)


args=parser.parse_args()

path=args.train_datapath #should have query target as subfolders
dataset_path_query = os.path.join(path, "query/")
dataset_path_target = os.path.join(path, "target/")

# print(os.listdir(dataset_path_target)[:5])
X=[] #this will contain 28,28 images

for img_name in sorted(os.listdir(dataset_path_target)):
    img = np.array(Image.open(os.path.join(dataset_path_target,img_name))) # 224,224 = 64 * 28,28
    sub_imgs=np.split(img,8)
    sub_imgs=[np.split(x_,8,axis=1) for x_ in sub_imgs]
    sub_imgs=np.array(sub_imgs) # 8,8,28,28
    sub_imgs=sub_imgs.reshape((-1,28,28))
    X.append(sub_imgs)

X=np.array(X)
X_target=X.reshape((-1,28,28))
np.save(args.target_array_file, X_target)

# print(os.listdir(dataset_path_target)[:5])
X=[] #this will contain 28,28 images
for img_name in sorted(os.listdir(dataset_path_query)):
    img = np.array(Image.open(os.path.join(dataset_path_query,img_name))) # 224,224 = 64 * 28,28
    sub_imgs=np.split(img,8)
    sub_imgs=[np.split(x_,8,axis=1) for x_ in sub_imgs]
    sub_imgs=np.array(sub_imgs) # 8,8,28,28
    sub_imgs=sub_imgs.reshape((-1,28,28))
    X.append(sub_imgs)

X=np.array(X)
X_query=X.reshape((-1,28,28))
np.save(args.query_array_file, X_query)

X_combined_query_target = np.concatenate((X_query, X_target))
print(X_combined_query_target.shape)
np.save(args.query_target_array_file, X_combined_query_target)

