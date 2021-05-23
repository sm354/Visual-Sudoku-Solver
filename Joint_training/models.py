import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

class RRN(nn.Module):
    def __init__(self, embed_dim=16, sudoku_cells=8, hidden_dim=96, num_steps=32, device='cpu'):
        # sudoku_cells means we will have sudoku_cells x sudoku_cells in the sudoku table
        super(RRN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.device = device
        
        ############################ FOR MESSAGE SIGNALS
        # find the required edges in the graph to have communication of message signals
        indices_of_cells=np.arange(0,sudoku_cells*sudoku_cells).reshape((sudoku_cells,sudoku_cells))
        edges_row, edges_col, edges_in_2x4=[],[],[]
        for i in range(8):
            vector = indices_of_cells[i,:]
            edges_row += [(i,j) for i in vector for j in vector if i!=j]
            vector = indices_of_cells[:,i]
            edges_col += [(i,j) for i in vector for j in vector if i!=j]
        for i in range(4):
            for j in range(2):
                vector = indices_of_cells[2*i:2*(i+1),4*j:4*(j+1)].reshape(-1)
                edges_in_2x4 += [(i,j) for i in vector for j in vector if i!=j]
        self.edges = torch.tensor(list(set(edges_row + edges_col + edges_in_2x4))).long().to(self.device)
        # self.edges contains all the possible pairs of communication between the cells of sudoku
        ############################
        

        ############################ FOR EMBEDDING ROW, COL INFORMATION
        # create row and col labels for the cells of sudoku table
        row_col = []
        for i in range(sudoku_cells):
            for j in range(sudoku_cells):
                row_col.append((i,j))
        self.row_col = torch.tensor(row_col).long().to(self.device)
        ############################
        

        ############################ EMBEDDING LAYERS
        # embedding the cell content {0,1,2,...,sudoku_cells}, row and column information for each cell in sudoku
        self.embed_dim = embed_dim
        # embed_1_init = torch.rand(sudoku_cells+1, self.embed_dim).to(self.device) #sudoku_cells+1 because possible digits in input : 0,1,2,3,...,sudoku_cells
        # self.embed_1 = nn.Embedding.from_pretrained(embed_1_init, freeze=False) #nn.Linear(sudoku_cells+1, self.embed_dim)
        # embed_2_init = torch.rand(sudoku_cells, self.embed_dim).to(self.device)
        # self.embed_2 = nn.Embedding.from_pretrained(embed_2_init, freeze=False) #nn.Linear(sudoku_cells, self.embed_dim)
        # embed_3_init = torch.rand(sudoku_cells, self.embed_dim).to(self.device)
        # self.embed_3 = nn.Embedding.from_pretrained(embed_3_init, freeze=False) #nn.Linear(sudoku_cells, self.embed_dim)
        ############################


        ############################ MLPs
        self.embeds_to_x = MLP_for_RRN(3*embed_dim, hidden_dim)
        self.message_mlp = MLP_for_RRN(2*hidden_dim, hidden_dim)
        self.mlp_for_lstm_inp = MLP_for_RRN(2*hidden_dim, hidden_dim)
        self.r_to_o_mlp = nn.Linear(hidden_dim, sudoku_cells+1) # only one linear layer as given in architecture details
        ############################

        # LSTM for looping over time i.e. num_steps
        self.LSTM = nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim) # since x and m will be concatentated and fed into lstm; x and m are of shape : batch_size*8*8, hidden_dim
        
        
    def forward(self, inp, y_true=None, loss_fn=None): # inp.shape=batch_size,9*9
        bs = inp.shape[0]
        inp = inp.view(-1)

        # embed the cell content
        inp = F.one_hot(inp, self.embed_dim).float()
        embedded_inp = inp #self.embed_1(inp) # batch_size*8*8, embed_dim
        # now also get row and column info of each cell embedded
        row_col = self.row_col.repeat(bs, 1)
        inp_row = F.one_hot(row_col[:,0], self.embed_dim).float()
        embedded_row = inp_row #self.embed_2(row_col[:,0])
        inp_col = F.one_hot(row_col[:,1], self.embed_dim).float()
        embedded_col = inp_col #self.embed_3(row_col[:,1])
        
        embedded_all = torch.cat([embedded_inp,embedded_row,embedded_col], dim=1)
        x = self.embeds_to_x(embedded_all) # batch_size*8*8, hidden_dim
        assert x.shape[1] == self.hidden_dim
        
        # x will be concatenated with m and then fed into LSTM
        # find message signals : over time i.e. num_steps
        # m_{i,j}^{t} = MLP(h_{i}^{t-1}, h_{j}^{t-1} 
        # since m^t requires h^{t-1}, remember past h, c
        h_for_msgs = x.detach().clone().to(self.device)
        l = 0
        for t in range(self.num_steps):
            h_for_msgs = h_for_msgs.view(-1, 64, self.hidden_dim)
            inp_for_msgs = h_for_msgs[:,self.edges].view(-1, 2*self.hidden_dim)
            msgs = self.message_mlp(inp_for_msgs).view(bs, -1, self.hidden_dim)
            
            # now sum up the message signals appropriately
            final_msgs = torch.zeros(h_for_msgs.shape).to(self.device)
            indices = self.edges[:,1].to(self.device)
            final_msgs = final_msgs.index_add(1, indices, msgs) # shape : batch_size, 64, self.hidden_dim
            final_msgs = final_msgs.view(-1, self.hidden_dim)
                        
            inp_to_lstm = self.mlp_for_lstm_inp(torch.cat([final_msgs,x],dim=1))
            h, c = self.LSTM(inp_to_lstm, (h,c)) if t!=0 else self.LSTM(inp_to_lstm, (torch.zeros(x.shape).to(self.device),torch.zeros(x.shape).to(self.device))) #x.detach().to(self.device)
            
            h_for_msgs = h
            o = self.r_to_o_mlp(c)
            
            l += loss_fn(o,y_true.long())

        out = o
        return (out,l) # out.shape = num_steps, batch_size*8*8, 9 : last dim is without-softmax over sudoku_cells(9)