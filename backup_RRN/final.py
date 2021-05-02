import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt

class MLP_for_RRN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_for_RRN, self).__init__()
        self.fc1=nn.Linear(input_dim, output_dim)
        self.fc2=nn.Linear(output_dim, output_dim)
        self.fc3=nn.Linear(output_dim, output_dim)
        # self.fc4=nn.Linear(output_dim, output_dim)
    
    def forward(self, inp):
        out = F.relu(self.fc1(inp))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        # out = self.fc4(out)
        return out

def compute_cp(pred, true): # pred and true are of shape = batch,81
    # number of exactly correct predictions
    return torch.sum(torch.sum(pred==true,dim=1)==81)

def compute_micro_score(pred, true):
    # this finds percentage of correct predicited digits and then averaged over batch
    return torch.mean(100.*torch.sum(pred==true,dim=1)/81)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
embed_dim=16
sudoku_cells=9
hidden_dim=96
num_steps=32
batch_size=32


data_X=np.load('drive/My Drive/Colab Notebooks/COL870/sample_X.npy')[:1500]
data_Y=np.load('drive/My Drive/Colab Notebooks/COL870/sample_Y.npy')[:1500]

dataset=TensorDataset(torch.tensor(data_X[:1024]),torch.tensor(data_Y[:1024]))
trainloader=DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

dataset=TensorDataset(torch.tensor(data_X[1024:]),torch.tensor(data_Y[1024:]))
testloader=DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)




embeds_to_x = MLP_for_RRN(embed_dim, hidden_dim).to(device)
message_mlp = MLP_for_RRN(2*hidden_dim, hidden_dim).to(device)
mlp_for_lstm_inp = MLP_for_RRN(2*hidden_dim, hidden_dim).to(device)
r_to_o_mlp = nn.Linear(hidden_dim, sudoku_cells+1).to(device) # only one linear layer as given in architecture details
mlps = [embeds_to_x, message_mlp, mlp_for_lstm_inp, r_to_o_mlp]
LSTM = nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim).to(device)# since x and m will be concatentated and fed into lstm; x and m are of shape : batch_size*9*9, hidden_dim


embed = torch.nn.functional.one_hot ## ALERT


optimizers = []
for mlp in mlps:
    optimizers.append(optim.Adam(mlp.parameters(), lr=2e-4, weight_decay=1e-4))
optimizers.append(optim.Adam(LSTM.parameters(), lr=2e-4, weight_decay=1e-4))
loss_fn = nn.CrossEntropyLoss()




############################ get edges
# find the required edges in the graph to have communication of message signals
indices_of_cells=np.arange(0,sudoku_cells*sudoku_cells).reshape((sudoku_cells,sudoku_cells))
edges_row, edges_col, edges_in_3x3=[],[],[]
for i in range(9):
    vector = indices_of_cells[i,:]
    edges_row += [(i,j) for i in vector for j in vector if i!=j]
    vector = indices_of_cells[:,i]
    edges_col += [(i,j) for i in vector for j in vector if i!=j]
for i in range(3):
    for j in range(3):
        vector = indices_of_cells[3*i:3*(i+1),3*j:3*(j+1)].reshape(-1)
        edges_in_3x3 += [(i,j) for i in vector for j in vector if i!=j]

edges = list(set(edges_row + edges_col + edges_in_3x3))
# edges = [ (i + (b*81), j + (b*81)) for b in range(batch_size) for i,j in edges]
edges = torch.tensor(edges).long().to(device)


# def cross(a):
#     return [(i, j) for i in a.flatten() for j in a.flatten() if not i == j]

# idx = np.arange(81).reshape(9, 9)
# rows, columns, squares = [], [], []
# for i in range(9):
#     rows += cross(idx[i, :])
#     columns += cross(idx[:, i])
# for i in range(3):
#     for j in range(3):
#         squares += cross(idx[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3])

# edges_base = list(set(rows + columns + squares))
# batched_edges = [(i + (b * 81), j + (b * 81)) for b in range(batch_size) for i, j in edges_base]
# edges = torch.Tensor(batched_edges).long()#.to(device)



# self.edges contains all the possible pairs of communication between the cells of sudoku
############################


num_epochs=100
for epoch in range(num_epochs):
    lss = 0
        
    total, correct, micro_score = 0, 0, 0

    for batch_id, (X,Y) in enumerate(trainloader):
        if X.shape[0]!=batch_size:
            continue


        X = X.flatten()
        Y = Y.flatten()
        if epoch == 0 and batch_id == 0:
            print('printing X shape: ', X.shape)
        
        X = embed(X.long(), embed_dim).float()
        
        if epoch == 0 and batch_id == 0:
            print('printing embedded X shape: ', X.shape)


        X = X.to(device)
        Y = Y.to(device)

        X = embeds_to_x(X)

        H = X.detach().clone().to(device)

        for optimizer in optimizers:
            optimizer.zero_grad()

        loss = 0
        HiddenState,CellState = torch.zeros(X.shape).to(device), torch.zeros(X.shape).to(device)

        for i in range(num_steps):

            n_nodes = H.shape[0]
            n_edges = edges.shape[0]
            n_embed = H.shape[1]
            assert n_embed == 96

            H = H.view(-1,81,96)

            assert H.shape[0] == 32

            message_inputs = H[:,edges]
            message_inputs = message_inputs.view(-1, 2*96)

            messages = message_mlp(message_inputs).view(H.shape[0],-1,96)

            updates = torch.zeros(H.shape).to(device)
            idx_j = edges[:, 1].to(device)
            H = updates.index_add(1, idx_j, messages)

            H = H.view(-1,96)

            H = mlp_for_lstm_inp(torch.cat([H, X], dim=1))
            HiddenState, CellState = LSTM(H, (HiddenState, CellState))

            H = CellState ## ALERT

            Y_pred = r_to_o_mlp(H)

            loss += loss_fn(Y_pred, Y.long())

        loss /= batch_size

        loss.backward()
        for optimizer in optimizers:
            optimizer.step()


        lss += loss.item()

        Y_pred = Y_pred.argmax(dim=1).view(-1,81)
        Y = Y.view(-1,81)

        correct_predictions = compute_cp(Y_pred.cpu(),Y.cpu()) # number of exactly correct predictions
        correct += correct_predictions
        total += Y.shape[0]

        micro_correct_digits = compute_micro_score(Y_pred.cpu(),Y.cpu()) # this finds percentage of correct predicited digits and then averaged over batch
        micro_score += micro_correct_digits
        
    micro_score /= batch_id
    lss /= batch_id

    print("epoch:",epoch,"|loss:",lss,"| Completely correct predictions:",100.*correct/total,"| Percentage of Correctly predicted digits:",micro_score)
