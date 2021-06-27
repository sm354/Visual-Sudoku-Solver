import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from rrn2 import *
import os

import os
import matplotlib.pyplot as plt
import PIL.Image as Image
import torch
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn import decomposition
import torch

from classifier_arabicmnist import *

torch.manual_seed(0)

parser = argparse.ArgumentParser(description='Testing Recurrent Relational Network on Sudoku')
parser.add_argument('--data_dir_query_images',type=str)
parser.add_argument('--pretr_classifier',type=str)

parser.add_argument('--batch_size',type=int,default=64,help='give batch size')
parser.add_argument('--model_path',type=str,help='load model')
parser.add_argument('--output_csv',type=str,help='path to output csv')
parser.add_argument('--num_steps',type=int,default=20,help='number of steps in RRN')
args=parser.parse_args()

device='cuda:0' if torch.cuda.is_available() else 'cpu'


def predict(model, device, test_loader, use_cuda):
    model.eval()
    predictions = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data.float())
            # print(output.shape)
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            pred = output # of shape(batch, 9)
            # predictions.extend(pred.tolist())
            predictions.append(pred.cpu())

    # print(predictions)

    return np.concatenate(predictions) # of shape (nb,9)





#TESTING QUERY
X=[] #this will contain 28,28 images

# i = 0
for img_name in sorted(os.listdir(args.data_dir_query_images)):
    # i+=1
    # if(i ==10):
    #     break
    img = np.array(Image.open(os.path.join(args.data_dir_query_images,img_name))) # 224,224 = 64 * 28,28
    sub_imgs=np.split(img,8)
    sub_imgs=[np.split(x_,8,axis=1) for x_ in sub_imgs]
    sub_imgs=np.array(sub_imgs) # 8,8,28,28
    sub_imgs=sub_imgs.reshape((-1,28,28))
    X.append(sub_imgs)

X=np.array(X)
X_input_query=X.reshape((-1,28,28))
print(X_input_query.shape)


X_input_query = -1*((X_input_query)/255. -1.) #for making it a sparse matrix
# shape is b, 28, 28

classifier = LeNet(9).to(device)
classifier.load_state_dict(torch.load(args.pretr_classifier,map_location=device))
# classifier = torch.load(args.pretr_classifier, map_location=device)
classifier.eval()

sudoku_cells = 8
# #load target dataset data
batchsize_fulldata = args.batch_size
fulldata_loader = DataLoader(torch.from_numpy(X_input_query.reshape(-1, 1, 28, 28)).to(device), batch_size=batchsize_fulldata, shuffle=False)

fulldata_labels = predict(classifier, device, fulldata_loader, True) #shape (nb,9)
print(fulldata_labels.shape)
fulldata_labels = fulldata_labels.reshape(-1, sudoku_cells*sudoku_cells, 9) #shape (b, 64)
print('fulldata shape after classifying = ', fulldata_labels.shape)






embed_dim=16
sudoku_cells=8
hidden_dim=96
model = RRN(embed_dim=embed_dim, sudoku_cells=sudoku_cells, hidden_dim=hidden_dim, num_steps=args.num_steps, device=device)
model = model.to(device)
model.load_state_dict(torch.load(args.model_path,map_location=device))
print('RRN model loaded')



# X = np.load(args.data_dir).astype(np.uint8).reshape((-1,8*8))
X = fulldata_labels
dataset = TensorDataset(torch.tensor(X))
data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)


Y_pred=[]
model.eval()
with torch.no_grad():
    for batch_id, X in enumerate(data_loader):
        # if X.shape[0] != batch_size:
            # continue
        X = X[0].to(device).long()
        Y_ = model(X,training=False)
        Y_predicted = Y_.argmax(dim=1)
        Y_predicted = Y_predicted.view(-1,8*8)
        Y_pred.append(Y_predicted)

Y_pred=torch.cat(Y_pred).cpu().numpy().reshape((-1,64))
df=pd.DataFrame(Y_pred)
df.insert(0,"",['%i.png'%i for i in range(Y_pred.shape[0])])
df.to_csv(args.output_csv,header=False,index=False)