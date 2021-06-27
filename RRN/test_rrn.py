import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from part2_rrn import *

torch.manual_seed(0)

parser = argparse.ArgumentParser(description='Testing Recurrent Relational Network on Sudoku')
parser.add_argument('--data_dir',type=str)
parser.add_argument('--batch_size',type=int,default=64,help='give batch size')
parser.add_argument('--model_path',type=str,help='load model')
parser.add_argument('--output_csv',type=str,help='path to output csv')
parser.add_argument('--num_steps',type=int,default=20,help='number of steps in RRN')
args=parser.parse_args()

device='cuda:0' if torch.cuda.is_available() else 'cpu'
embed_dim=16
sudoku_cells=8
hidden_dim=96
model = RRN(embed_dim=embed_dim, sudoku_cells=sudoku_cells, hidden_dim=hidden_dim, num_steps=args.num_steps, device=device)
model = model.to(device)
model.load_state_dict(torch.load(args.model_path,map_location=device))
print('RRN model loaded')

X = np.load(args.data_dir).astype(np.uint8).reshape((-1,8*8))
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