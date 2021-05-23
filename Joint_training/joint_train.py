import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import argparse
from rrn import *
from classifier_arabicmnist import *

def is_set_correct(array):
    # print(array)
    # print(set(array))
    if len(set(array)) >= 8:
        return True
    return False


def board_accuracy(labels):
    #labels are of shape (totalsmall sudokus, 8, 8,)
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
        j, k = (i // 4) * 4, (i % 2) * 2
        # if(np.all(np.apply_along_axis(is_set_correct, axis = 1, arr = labels[:, :, i])) == True or np.all(np.apply_along_axis(is_set_correct, axis = 1, arr = labels[:, i, :])) == True or np.all(np.apply_along_axis(is_set_correct, axis = 1, arr = labels[:, k:k+2, j:j+4].reshape(-1, 8))) !=True ):
        #   correct+=1
        # total+=1

        arr1 = np.apply_along_axis(is_set_correct, axis = 1, arr = labels[:, :, i])
        arr2 = np.apply_along_axis(is_set_correct, axis = 1, arr = labels[:, i, :])
        arr3 = np.apply_along_axis(is_set_correct, axis = 1, arr = labels[:, k:k+2, j:j+4].reshape(-1, 8))
        arr = arr1*arr2*arr3
        assert(arr.shape[0] == labels.shape[0] and len(arr.shape) == 1)
        final_bool_arr *= arr
        subatomic_correct += arr1.sum() + arr2.sum() + arr3.sum()
        # print(subatomic_correct)

        # if len(set(labels[i,:])) != 9 or len(set(grid[:,i])) != 9 or len(set(grid[j:j+3, k:k+3].ravel())) != 9:
        #   return False
    return final_bool_arr.sum()/final_bool_arr.shape[0], subatomic_correct/(3*8*labels.shape[0])

def compute_acc_sudoku(pred, true): # pred and true are of shape = batch,64
    return torch.sum(torch.sum(pred==true,dim=1)==64) # number of exactly correct predictions

def compute_acc_digit(pred, true):
    temp = 100.*torch.sum(pred==true,dim=1).float()/64.
    return torch.sum(temp)/temp.shape[0] # this finds percentage of correct predicited digits and then averaged over batch

def save_plots(train_cp, val_cp, train_micro, val_micro, plts_dir, resume=False):
    if resume!=False:
        data=np.load(plts_dir+'.npy')
        train_cp=list(data[0])+train_cp
        val_cp=list(data[1])+val_cp
        train_micro=list(data[2])+train_micro
        val_micro=list(data[3])+val_micro
    plt.title('Sudoku Level Accuracy')
    plt.xlabel('epochs')
    plt.plot(train_cp,label='train_acc')
    plt.plot(val_cp,label='val_acc')
    plt.grid()
    plt.legend()
    plt.savefig(plts_dir+'_sud.png')
    plt.title('Digit Level Accuracy')
    plt.xlabel('epochs')
    plt.plot(train_micro,label='train_acc')
    plt.plot(val_micro,label='val_acc')
    plt.grid()
    plt.legend()
    plt.savefig(plts_dir+'_dig.png')
    np.save(plts_dir+'.npy', np.array([train_cp,val_cp,train_micro,val_micro]))


parser = argparse.ArgumentParser(description='Training Recurrent Relational Network on Sudoku')
parser.add_argument('--data_dir',type=str)
parser.add_argument('--pretr_classifier',type=str)
parser.add_argument('--loss_reg', type = str)

parser.add_argument('--lr_classifier', type=float, default = 1e-4)
parser.add_argument('--lr_rrn', type=float, default = 2e-3)

parser.add_argument('--num_epochs',type=int,default=100,help='number of epochs of training')
parser.add_argument('--num_steps',type=int,default=32,help='number of steps in RRN')
parser.add_argument('--batch_size',type=int,default=64,help='give batch size (of optimization algo)')
parser.add_argument('--resume',default=False,help='give argument if training has to be resumed')
parser.add_argument('--exp_name',type=str,default=False,help='experiment name')
parser.add_argument('--savemodel',type=str,default=False,help='give directory to save models if model is to be saved')
parser.add_argument('--saveplot',type=str,default=False,help='give directory to save plots if plots to be plotted name')
args=parser.parse_args()
print(args)

batch_size=args.batch_size
num_steps=args.num_steps
embed_dim=16
sudoku_cells=8
hidden_dim=96
device='cuda:0' if torch.cuda.is_available() else 'cpu'

data=np.load(args.data_dir).astype(np.uint8).reshape((-1,sudoku_cells*sudoku_cells, 28, 28))/255. #shaope is (b, 64, 28, 28)
# print(data)
X_data=data[:10000]
Y_data=data[10000:]


################################################################################
#Find true noisy labels using the classifier, X_data_truenoisy, Y_data_truenoisy
classifier = torch.load(args.pretr_classifier, map_location=device)
classifier.eval()

#load target dataset data
batchsize_fulldata = 64
print(device)
fulldata_loader = DataLoader(torch.from_numpy(data.reshape(-1, 1, 28, 28)).to(device), batch_size=batchsize_fulldata, shuffle=False)

fulldata_labels = predict(classifier, device, fulldata_loader, True)
fulldata_labels = fulldata_labels.reshape(-1, sudoku_cells*sudoku_cells) #shape (b, 64)
print('fulldata shape after classifying = ', fulldata_labels.shape)

X_data_truenoisy = torch.from_numpy(fulldata_labels[:10000])
Y_data_truenoisy = torch.from_numpy(fulldata_labels[10000:])

# deallocate memory
del fulldata_labels
del fulldata_loader

#now we will use classifier in train mode
classifier.train
###################################################################################

split = 0.90
X_train=X_data[:int(split*X_data.shape[0])]
Y_train=Y_data[:int(split*Y_data.shape[0])]
X_train_truenoisy=X_data_truenoisy[:int(split*X_data_truenoisy.shape[0])]
Y_train_truenoisy=Y_data_truenoisy[:int(split*Y_data_truenoisy.shape[0])]
print(X_train.shape, Y_train.shape, X_train_truenoisy.shape, Y_train_truenoisy.shape)
train_dataset=TensorDataset(torch.tensor(X_train),torch.tensor(Y_train), X_train_truenoisy, Y_train_truenoisy)
train_data_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

X_val=X_data[int(split*X_data.shape[0]):]
Y_val=Y_data[int(split*Y_data.shape[0]):]
X_val_truenoisy=X_data_truenoisy[int(split*X_data_truenoisy.shape[0]):]
Y_val_truenoisy=Y_data_truenoisy[int(split*Y_data_truenoisy.shape[0]):]
val_dataset=TensorDataset(torch.tensor(X_val),torch.tensor(Y_val), X_val_truenoisy, Y_val_truenoisy)
val_data_loader=DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

print('train and val dataset prepared')

model = RRN(embed_dim=embed_dim, sudoku_cells=sudoku_cells, hidden_dim=hidden_dim, num_steps=num_steps, device=device)
model = model.to(device)
if args.resume!=False:
    model.load_state_dict(torch.load(args.savemodel+args.exp_name+'_rrn.pth',map_location=device))
    classifier = LeNet(9).to(device)
    classifier.load_state_dict(torch.load(args.savemodel+args.exp_name+'_classifier.pth',map_location=device))
print('RRN model loaded')
print(model)

#######for rrn
# optimizer=torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)
optimizer=torch.optim.Adam(model.parameters(), lr=args.lr_rrn, weight_decay=1e-4)

loss_fn=nn.CrossEntropyLoss()
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

#######for classifier
optimizer_classifier=torch.optim.Adam(classifier.parameters(), lr=args.lr_classifier)
# optimizer_classifier=torch.optim.Adam(classifier.parameters(), lr=1e-3)

loss_fn_classifier=nn.CrossEntropyLoss() #check mse also


print('beginning training')
train_loss,val_loss=[],[]
train_cp,val_cp=[],[]
train_micro,val_micro=[],[]

classifier_trainloss_inputreg, classifier_valloss_inputreg = [], []
classifier_trainloss_outputreg, classifier_valloss_outputreg = [], []
totalloss = []

correct_trainboards_acc = []
correct_train8level_acc = []
correct_valboards_acc = []
correct_val8level_acc = []

num_epochs=args.num_epochs
for epoch in range(num_epochs):
    print("\n\n\nepoch:",epoch)
    #training
    model.train()
    classifier.train()

    lss=0
    lss_input_reg = 0
    lss_output_reg = 0
    lss_total = 0

    total, correct, micro_score = 0, 0, 0
    acc_boards, acc_8level,  = 0, 0 #for rnn output
    acc_boards_classifier, acc_8level_classifier = 0, 0

    for batch_id, (X_img, Y_img, X_truenoisy, Y_truenoisy) in enumerate(train_data_loader):
        if X_img.shape[0] != batch_size: # probably useful for avoiding exploding/vanishing gradients
            continue
        
        X_img, Y_img = X_img.to(device), Y_img.to(device)
        X_truenoisy, Y_truenoisy = X_truenoisy.to(device), Y_truenoisy.to(device)

        X = classifier(X_img.reshape(-1, 1, 28, 28).float()).reshape(X_img.shape[0], sudoku_cells*sudoku_cells, -1) # (b, 8*8, 9)
        Y = classifier(Y_img.reshape(-1, 1, 28, 28).float()).reshape(X_img.shape[0], sudoku_cells*sudoku_cells, -1) # (b, 8*8, 9)
        # print(X.shape)
        Y_onehot = Y.argmax(dim=2) #shape (b,8*8)
        X_onehot = X.argmax(dim=2) #shape (b,8*8)


        Y_onehot = Y_onehot.view(-1)
        optimizer.zero_grad()
        optimizer_classifier.zero_grad()
        
        Y_, l = model(X,Y_onehot,loss_fn)
        Y_pred = Y_.argmax(dim=1)
        l /= batch_size

        # losses for regularization
        l_input_reg = loss_fn_classifier(X.reshape(-1, X.shape[2]), X_truenoisy.reshape(-1).long())
        l_output_reg = loss_fn_classifier(Y.reshape(-1, Y.shape[2]), Y_truenoisy.reshape(-1).long())

        if(args.loss_reg == 'yes' and epoch<5): #epoch condition removes the regularization once the epoch has been reached
            total_loss = l+l_input_reg+l_output_reg
        
        else:
            if(epoch == 5):
                print('**********************NOW USING UNREGULARIZED LOSS**************************')
            total_loss = l

        total_loss.backward()
        optimizer.step()
        optimizer_classifier.step()

        # l.backward()
        # optimizer.step()

        Y_pred = Y_pred.view(-1,sudoku_cells*sudoku_cells)
        Y_onehot = Y_onehot.view(-1,sudoku_cells*sudoku_cells)
        assert X.shape[0]==Y_onehot.shape[0]
        lss += l.item()
        lss_input_reg += l_input_reg.item()
        lss_output_reg += l_output_reg.item()
        lss_total += total_loss.item()

        correct_predictions = compute_acc_sudoku(Y_pred.cpu(),Y_onehot.long().cpu()).item() # number of exactly correct predictions
        correct += correct_predictions
        total += Y_onehot.shape[0]
        micro_correct_digits = compute_acc_digit(Y_pred.cpu(),Y_onehot.long().cpu()) # this finds percentage of correct predicited digits and then averaged over batch
        micro_score += micro_correct_digits

        # Y_pred is our predicted labels for which we can calculate if boards and 8levels are correct or not
        acc1, acc2 = board_accuracy(Y_pred.cpu().reshape(-1, sudoku_cells, sudoku_cells))
        acc_boards += acc1
        acc_8level += acc2

        acc1_, acc2_ = board_accuracy(Y_onehot.cpu().reshape(-1, sudoku_cells, sudoku_cells))
        acc_boards_classifier += acc1_
        acc_8level_classifier += acc2_

    lss /= batch_id
    lss_input_reg, lss_output_reg, lss_total = lss_input_reg/batch_id, lss_output_reg/batch_id, lss_total/batch_id
    micro_score /= batch_id
    acc_boards, acc_8level = acc_boards/batch_id, acc_8level/batch_id
    acc_boards_classifier, acc_8level_classifier = acc_boards_classifier/batch_id, acc_8level_classifier/batch_id


    print("---------TRAIN-----------")
    print("train loss (rnn, reg_input, reg_output, total):(",lss, lss_input_reg, lss_output_reg, lss_total, ")| train acc sudoku:",100.*correct/total,"| train acc digit:",micro_score)
    print("-----------rnn output constraints satisfied----------")
    print("train boards correct = ", acc_boards, "train 8level correct", acc_8level)
    print("-----------Classifier output constraints satisfied----------")
    print("train boards correct = ", acc_boards_classifier, "train 8level correct", acc_8level_classifier)
    train_loss.append(lss)
    classifier_trainloss_inputreg.append(lss_input_reg)
    classifier_trainloss_outputreg.append(lss_output_reg)
    totalloss.append(lss_total)

    correct_trainboards_acc.append(acc_boards)
    correct_train8level_acc.append(acc_8level)

    train_cp.append(100.*correct/total)
    train_micro.append(micro_score)

    #validation
    model.eval()
    classifier.eval()

    lss=0
    lss_input_reg = 0
    lss_output_reg = 0
    lss_total = 0

    total,correct,micro_score = 0,0,0
    acc_boards, acc_8level,  = 0, 0
    acc_boards_classifier, acc_8level_classifier = 0, 0

    with torch.no_grad():
        for batch_id, (X_img, Y_img, X_truenoisy, Y_truenoisy) in enumerate(val_data_loader):
            if X_img.shape[0] != batch_size:
                continue

            X_img, Y_img = X_img.to(device), Y_img.to(device)
            X_truenoisy, Y_truenoisy = X_truenoisy.to(device), Y_truenoisy.to(device)

            X = classifier(X_img.reshape(-1, 1, 28, 28).float()).reshape(X_img.shape[0], sudoku_cells*sudoku_cells, -1) # (b, 8*8, 9)
            Y = classifier(Y_img.reshape(-1, 1, 28, 28).float()).reshape(X_img.shape[0], sudoku_cells*sudoku_cells, -1) # (b, 8*8, 9)
            # print(X.shape)
            Y_onehot = Y.argmax(dim=2) #shape (b,8*8)
            X_onehot = X.argmax(dim=2) #shape (b,8*8)

            Y_onehot = Y_onehot.view(-1)
            
            Y_, l = model(X,Y_onehot,loss_fn)
            Y_pred = Y_.argmax(dim=1)
            l /= batch_size

            # losses for regularization
            l_input_reg = loss_fn_classifier(X.reshape(-1, X.shape[2]), X_truenoisy.reshape(-1).long())
            l_output_reg = loss_fn_classifier(Y.reshape(-1, Y.shape[2]), Y_truenoisy.reshape(-1).long())

            if(args.loss_reg == 'yes' and epoch<5):
                total_loss = l+l_input_reg+l_output_reg
        
            else:
                if(epoch == 5):
                    print('**********************NOW USING UNREGULARIZED LOSS**************************')
                total_loss = l

            Y_pred = Y_pred.view(-1,sudoku_cells*sudoku_cells)
            Y_onehot = Y_onehot.view(-1,sudoku_cells*sudoku_cells)
            lss += l.item()
            lss_input_reg += l_input_reg.item()
            lss_output_reg += l_output_reg.item()
            lss_total += total_loss.item()

            correct_predictions = compute_acc_sudoku(Y_pred.cpu(),Y_onehot.long().cpu()).item() # number of exactly correct predictions
            correct += correct_predictions
            total += Y_onehot.shape[0]
            micro_correct_digits = compute_acc_digit(Y_pred.cpu(),Y_onehot.long().cpu()) # this finds percentage of correct predicited digits and then averaged over batch
            micro_score += micro_correct_digits

            # Y_pred is our predicted labels for which we can calculate if boards and 8levels are correct or not
            acc1, acc2 = board_accuracy(Y_pred.cpu().reshape(-1, sudoku_cells, sudoku_cells))
            acc_boards += acc1
            acc_8level += acc2

            acc1_, acc2_ = board_accuracy(Y_onehot.cpu().reshape(-1, sudoku_cells, sudoku_cells))
            acc_boards_classifier += acc1_
            acc_8level_classifier += acc2_

    lss /= batch_id
    lss_input_reg, lss_output_reg, lss_total = lss_input_reg/batch_id, lss_output_reg/batch_id, lss_total/batch_id
    micro_score /= batch_id
    acc_boards, acc_8level = acc_boards/batch_id, acc_8level/batch_id
    acc_boards_classifier, acc_8level_classifier = acc_boards_classifier/batch_id, acc_8level_classifier/batch_id

    print("\n---------VALIDATION-----------")
    print("valid loss (rnn, reg_input, reg_output, total):(",lss, lss_input_reg, lss_output_reg, lss_total, ")| valid acc sudoku:",100.*correct/total,"| valid acc digit:",micro_score)
    print("-----------rnn output constraints satisfied----------")
    print("val boards correct = ", acc_boards, "val 8level correct", acc_8level)
    print("-----------Classifier output constraints satisfied----------")
    print("val boards correct = ", acc_boards_classifier, "val 8level correct", acc_8level_classifier)

    val_loss.append(lss)
    classifier_valloss_inputreg.append(lss_input_reg)
    classifier_valloss_outputreg.append(lss_output_reg)
    totalloss.append(lss_total)

    correct_valboards_acc.append(acc_boards)
    correct_val8level_acc.append(acc_8level)

    val_cp.append(100.*correct/total)
    val_micro.append(micro_score)
    
    if epoch%10==0: #save model every 10 epochs
        torch.save(model.state_dict(),args.savemodel+args.exp_name+'_rrn.pth')
        torch.save(classifier.state_dict(),args.savemodel+args.exp_name+'_classifier.pth')
    
    # scheduler.step(lss_total)

if args.saveplot!=False:
    save_plots(train_cp, val_cp, train_micro, val_micro, args.saveplot+args.exp_name, args.resume)