import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt

def compute_cp(pred, true): # pred and true are of shape = batch,81
	# number of exactly correct predictions
	return torch.sum(torch.sum(pred==true,dim=1)==81)

def compute_micro_score(Y_pred, Y):
	# this finds percentage of correct predicited digits and then averaged over batch
	return torch.mean(100.*torch.sum(pred==true,dim=1)/81)

batch_size=32
embed_dim=16
sudoku_cells=9
hidden_dim=96
num_steps=32
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

data_X=np.load('drive/My Drive/Colab Notebooks/COL870/sample_X.npy')[:1024]
data_Y=np.load('drive/My Drive/Colab Notebooks/COL870/sample_Y.npy')[:1024]
dataset=TensorDataset(torch.tensor(data_X),torch.tensor(data_Y))
data_loader=DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

model = RRN(embed_dim=embed_dim, sudoku_cells=sudoku_cells, hidden_dim=hidden_dim, num_steps=num_steps, device=device)
model = model.to(device)
print(model)

optimizer=torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)
loss_fn=nn.CrossEntropyLoss()
# scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

num_epochs=100
model.train()
train_loss=[]
for epoch in range(num_epochs):
    lss=0
    total, correct, micro_score = 0, 0, 0
    for batch_id, (X,Y) in enumerate(data_loader):
        X, Y = X.to(device).long(), Y.to(device).long()
        X = X.view(-1,sudoku_cells*sudoku_cells)
        Y = Y.view(-1)
        
        optimizer.zero_grad()
        Y_ = model(X)
        
        l=0
        for i in range(num_steps):
            ls=loss_fn(Y_[i],Y)
            l+=ls

        Y_pred = Y_[-1].argmax(dim=1)

        l /= batch_size
        l.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), 5) # clip gradient to 5
        optimizer.step()
        
        Y_pred = Y_pred.view(-1,81)
        Y = Y.view(-1,81)
        assert X.shape[0]==Y.shape[0]

        lss += l.item()
        correct_predictions = compute_cp(Y_pred.cpu(),Y.cpu()) # number of exactly correct predictions
        correct += correct_predictions
        total += Y.shape[0]

        micro_correct_digits = compute_micro_score(Y_pred.cpu(),Y.cpu()) # this finds percentage of correct predicited digits and then averaged over batch
        micro_score += micro_correct_digits
        
    lss /= batch_id
    micro_score /= batch_id
    print("epoch:",epoch,"|	loss:",lss,"| Completely correct predictions:",100.*correct/total,"| Percentage of Correctly predicted digits:",micro_score)

torch.save(model.state_dict(),'RRN.pth')