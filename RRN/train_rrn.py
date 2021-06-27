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

def compute_acc_sudoku(pred, true): # pred and true are of shape = batch,64
    return torch.sum(torch.sum(pred==true,dim=1)==64) # number of exactly correct predictions

def compute_acc_digit(pred, true):
    temp = 100.*torch.sum(pred==true,dim=1).float()/64.
    return torch.sum(temp)/temp.shape[0] # this finds percentage of correct predicited digits and then averaged over batch

def is_set_correct(array):
    # print(array)
    # print(set(array))
    if len(set(array)) >= 8:
        return True
    return False

def board_accuracy(labels):
    #labels are of shape (totalsmall images in all sudoku which is divisible by 64,)
    # labels = labels.reshape((labels.shape[0]//64, -1))
    # labels = labels.reshape((-1, 8, 8))
    # print(labels.shape)
    # print(labels[0])
    # # print(labels[10000])

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

def save_plots(train_cp, val_cp, train_micro, val_micro, train_8level, val_8level, train_boardlevel, val_boardlevel, plts_dir, resume=False):
    if resume!=False:
        data=np.load(plts_dir+'.npy')
        train_cp=list(data[0])+train_cp
        val_cp=list(data[1])+val_cp
        train_micro=list(data[2])+train_micro
        val_micro=list(data[3])+val_micro
        
        train_8level=list(data[4])+train_8level
        val_8level=list(data[5])+val_8level
        train_boardlevel=list(data[6])+train_boardlevel
        val_boardlevel=list(data[7])+val_boardlevel

    plt.figure()
    plt.title('Sudoku Level Accuracy')
    plt.xlabel('epochs')
    plt.plot(train_cp,label='train_acc')
    plt.plot(val_cp,label='val_acc')
    plt.grid()
    plt.legend()
    plt.savefig(plts_dir+'_sud.png')
    plt.figure()
    plt.title('Digit Level Accuracy')
    plt.xlabel('epochs')
    plt.plot(train_micro,label='train_acc')
    plt.plot(val_micro,label='val_acc')
    plt.grid()
    plt.legend()
    plt.savefig(plts_dir+'_dig.png')

    plt.figure()
    plt.title('8 Level Accuracy')
    plt.xlabel('epochs')
    plt.plot(train_8level,label='train_acc')
    plt.plot(val_8level,label='val_acc')
    plt.grid()
    plt.legend()
    plt.savefig(plts_dir+'_8.png')
    plt.figure()
    plt.title('Board Level Accuracy')
    plt.xlabel('epochs')
    plt.plot(train_boardlevel,label='train_acc')
    plt.plot(val_boardlevel,label='val_acc')
    plt.grid()
    plt.legend()
    plt.savefig(plts_dir+'_board.png')
    np.save(plts_dir+'.npy', np.array([train_cp,val_cp,train_micro,val_micro,train_8level,val_8level,train_boardlevel,val_boardlevel]))

parser = argparse.ArgumentParser(description='Training Recurrent Relational Network on Sudoku')
parser.add_argument('--data_dir',type=str)
parser.add_argument('--num_epochs',type=int,default=25,help='number of epochs of training')
parser.add_argument('--num_steps',type=int,default=20,help='number of steps in RRN')
parser.add_argument('--batch_size',type=int,default=64,help='give batch size (of optimization algo)')
parser.add_argument('--resume',default=False,help='give argument if training has to be resumed')
parser.add_argument('--exp_name',type=str,default=False,help='experiment name')
parser.add_argument('--savemodel',type=str,default=False,help='give directory to save models if model is to be saved')
parser.add_argument('--saveplot',type=str,default=False,help='give directory to save plots if plots to be plotted name')
args=parser.parse_args()

batch_size=args.batch_size
num_steps=args.num_steps
embed_dim=16
sudoku_cells=8
hidden_dim=96
device='cuda:0' if torch.cuda.is_available() else 'cpu'

data=np.load(args.data_dir).astype(np.uint8).reshape((-1,sudoku_cells*sudoku_cells))
X_data=data[:10000]
Y_data=data[10000:]

split = 0.95
X_train=X_data[:int(split*X_data.shape[0])]
Y_train=Y_data[:int(split*Y_data.shape[0])]
train_dataset=TensorDataset(torch.tensor(X_train),torch.tensor(Y_train))
train_data_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

X_val=X_data[int(split*X_data.shape[0]):]
Y_val=Y_data[int(split*Y_data.shape[0]):]
val_dataset=TensorDataset(torch.tensor(X_val),torch.tensor(Y_val))
val_data_loader=DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

print('train and val dataset prepared')

model = RRN(embed_dim=embed_dim, sudoku_cells=sudoku_cells, hidden_dim=hidden_dim, num_steps=num_steps, device=device)
model = model.to(device)
if args.resume!=False:
    model.load_state_dict(torch.load(args.savemodel+args.exp_name+'.pth',map_location=device))
print('RRN model loaded')
print(model)

optimizer=torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)
loss_fn=nn.CrossEntropyLoss()
# scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

print('beginning training')
train_loss,val_loss=[],[]
train_cp,val_cp=[],[]
train_micro,val_micro=[],[]

train_8level,val_8level=[],[]
train_boardlevel,val_boardlevel=[],[]

num_epochs=args.num_epochs
for epoch in range(num_epochs):
    print("epoch:",epoch)
    #training
    model.train()
    lss=0
    total, correct, micro_score = 0, 0, 0

    acc_boards, acc_8level,  = 0, 0 #for rnn output
    
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
        correct_predictions = compute_acc_sudoku(Y_pred.cpu(),Y.long().cpu()).item() # number of exactly correct predictions
        correct += correct_predictions
        total += Y.shape[0]
        micro_correct_digits = compute_acc_digit(Y_pred.cpu(),Y.long().cpu()) # this finds percentage of correct predicited digits and then averaged over batch
        micro_score += micro_correct_digits

        acc1, acc2 = board_accuracy(Y_pred.cpu().reshape(-1, sudoku_cells, sudoku_cells))
        acc_boards += acc1
        acc_8level += acc2

    lss /= batch_id
    micro_score /= batch_id
    
    acc_boards, acc_8level = acc_boards/batch_id, acc_8level/batch_id
    
    print("train loss:",lss,"| train acc sudoku:",100.*correct/total,"| train acc digit:",micro_score,"| train acc board level:",acc_boards,"| train acc 8level:",acc_8level)
    train_loss.append(lss)
    train_cp.append(100.*correct/total)
    train_micro.append(micro_score)
    
    train_8level.append(acc_8level)
    train_boardlevel.append(acc_boards)

    #validation
    model.eval()
    lss=0
    total,correct,micro_score = 0,0,0
    
    acc_boards, acc_8level,  = 0, 0
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
            correct_predictions = compute_acc_sudoku(Y_pred.cpu(),Y.long().cpu()).item() # number of exactly correct predictions
            correct += correct_predictions
            total += Y.shape[0]
            micro_correct_digits = compute_acc_digit(Y_pred.cpu(),Y.long().cpu()) # this finds percentage of correct predicited digits and then averaged over batch
            micro_score += micro_correct_digits

            acc1, acc2 = board_accuracy(Y_pred.cpu().reshape(-1, sudoku_cells, sudoku_cells))
            acc_boards += acc1
            acc_8level += acc2
        
    acc_boards, acc_8level = acc_boards/batch_id, acc_8level/batch_id

    lss /= batch_id
    micro_score /= batch_id
    print("val loss:",lss,"| val acc board:",100.*correct/total,"| val acc digit:",micro_score,"| val acc board level:",acc_boards,"| val acc 8level:",acc_8level)
    val_loss.append(lss)
    val_cp.append(100.*correct/total)
    val_micro.append(micro_score)
    
    val_8level.append(acc_8level)
    val_boardlevel.append(acc_boards)

    # if epoch%10==0: #save model every 10 epochs
        # torch.save(model.state_dict(),args.savemodel+args.exp_name+'.pth')
    
    # scheduler.step(lss)

if args.savemodel!=False:
    torch.save(model.state_dict(),args.savemodel+args.exp_name+'.pth')

if args.saveplot!=False:
    save_plots(train_cp, val_cp, train_micro, val_micro, train_8level, val_8level, train_boardlevel, val_boardlevel, args.saveplot+args.exp_name, args.resume)


