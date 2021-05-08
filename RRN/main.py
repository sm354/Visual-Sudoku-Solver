import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import argparse
from models import *

def compute_cp(pred, true): # pred and true are of shape = batch,64
    # number of exactly correct predictions
    return torch.sum(torch.sum(pred==true,dim=1)==64)

def compute_micro_score(pred, true):
    # this finds percentage of correct predicited digits and then averaged over batch
    temp = 100.*torch.sum(pred==true,dim=1).float()/64.
    return torch.sum(temp)/temp.shape[0]

parser = argparse.ArgumentParser(description='Training Recurrent Relational Network on Sudoku')
parser.add_argument('--data_dir', type=str)
parser.add_argument('--num_epochs',type=int,default=100,help='number of epochs of training')
parser.add_argument('--num_steps',type=int,default=32,help='number of steps in RRN')
parser.add_argument('--batch_size',type=int,default=32,help='give batch size (of optimization algo)')
parser.add_argument('--save_plots',default=False,help='give argument if plots to be plotted')
parser.add_argument('--resume',default=False,help='give argument if training has to be resumed')
parser.add_argument('--model_file',type=str,default='./RRN.pth',help='give path to save model')
parser.add_argument('--message',type=str,default=False,help='give any message if it needs to be printed')
args=parser.parse_args()

if args.message!=False:
    print(args.message)

batch_size=args.batch_size
num_steps=args.num_steps
embed_dim=16
sudoku_cells=8
hidden_dim=96
device='cuda:0' if torch.cuda.is_available() else 'cpu'

data=np.load(args.data_dir).astype(np.uint8).reshape((-1,sudoku_cells*sudoku_cells))
X_data=data[10000:]
Y_data=data[:10000]

X_train=X_data[:int(0.8*X_data.shape[0])]
Y_train=Y_data[:int(0.8*Y_data.shape[0])]
train_dataset=TensorDataset(torch.tensor(X_train),torch.tensor(Y_train))
train_data_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

X_val=X_data[int(0.8*X_data.shape[0]):]
Y_val=Y_data[int(0.8*Y_data.shape[0]):]
val_dataset=TensorDataset(torch.tensor(X_val),torch.tensor(Y_val))
val_data_loader=DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

print('train and val dataset prepared')

model = RRN(embed_dim=embed_dim, sudoku_cells=sudoku_cells, hidden_dim=hidden_dim, num_steps=num_steps, device=device)
model = model.to(device)
if args.resume!=False:
    model.load_state_dict(torch.load(args.model_file,map_location=device))
print('RRN model loaded')
print(model)

optimizer=torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)
loss_fn=nn.CrossEntropyLoss()

train_loss,val_loss=[],[]
train_cp,val_cp=[],[]
train_micro,val_micro=[],[]
print('beginning training')
num_epochs=args.num_epochs
for epoch in range(num_epochs):
    print("epoch:",epoch)
    #training
    # model.train()
    lss=0
    total, correct, micro_score = 0, 0, 0
    for batch_id, (X,Y) in enumerate(train_data_loader):
        if X.shape[0] != batch_size: # probably useful for avoiding exploding/vanishing gradients
            continue
        X, Y = X.to(device).long(), Y.to(device)
        Y = Y.view(-1)
        optimizer.zero_grad()
        
        Y_, l = model(X,Y,loss_fn)
        Y_pred = Y_.argmax(dim=1)
        l /= batch_size
        l.backward()

        optimizer.step()
        Y_pred = Y_pred.view(-1,sudoku_cells*sudoku_cells)
        Y = Y.view(-1,sudoku_cells*sudoku_cells)
        assert X.shape[0]==Y.shape[0]
        lss += l.item()
        correct_predictions = compute_cp(Y_pred.cpu(),Y.long().cpu()).item() # number of exactly correct predictions
        correct += correct_predictions
        total += Y.shape[0]
        micro_correct_digits = compute_micro_score(Y_pred.cpu(),Y.long().cpu()) # this finds percentage of correct predicited digits and then averaged over batch
        micro_score += micro_correct_digits

    lss /= batch_id
    micro_score /= batch_id
    print("train loss:",lss,"| train cp:",100.*correct/total,"| train micro:",micro_score)
    train_loss.append(lss)
    train_cp.append(100.*correct/total)
    train_micro.append(micro_score)

    #validation
    # model.eval()
    lss=0
    total,correct,micro_score = 0,0,0
    with torch.no_grad():
        for batch_id, (X,Y) in enumerate(val_data_loader):
            if X.shape[0] != batch_size:
                continue
            X, Y = X.to(device).long(), Y.to(device)
            Y = Y.view(-1)
            
            Y_,l = model(X,Y,loss_fn)
            Y_pred = Y_.argmax(dim=1)

            l /= batch_size
            Y_pred = Y_pred.view(-1,sudoku_cells*sudoku_cells)
            Y = Y.view(-1,sudoku_cells*sudoku_cells)
            lss += l.item()
            correct_predictions = compute_cp(Y_pred.cpu(),Y.long().cpu()).item() # number of exactly correct predictions
            correct += correct_predictions
            total += Y.shape[0]
            micro_correct_digits = compute_micro_score(Y_pred.cpu(),Y.long().cpu()) # this finds percentage of correct predicited digits and then averaged over batch
            micro_score += micro_correct_digits

    lss /= batch_id
    micro_score /= batch_id
    print("val loss:",lss,"| val cp:",100.*correct/total,"| val micro:",micro_score)
    val_loss.append(lss)
    val_cp.append(100.*correct/total)
    val_micro.append(micro_score)
    
    torch.save(model.state_dict(),args.model_file)
    # scheduler.step(lss)