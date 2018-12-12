#####################################################################################################################
# Imports
#####################################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

#####################################################################################################################
# Simple LSTM classifier with only one layer
#####################################################################################################################

class ProgGen(nn.Module):
    
    def __init__(self, vocab_size, hidden_size, num_layers, dropout = 0.5):
        super(ProgGen, self).__init__()
        
        self.hidden_size = hidden_size
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        
        # LSTM Multilayer. Receives (Seq x Batch x Features), Outputs (Seq x Batch x Hidden Size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, dropout = dropout)

        # Receives (Seq*Batch x Hidden Size), outputs (Seq*Batch x input_size)
        self.fc1 = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, idxs):
        x = self.embed(idxs)
        x, _ = self.lstm(x)
        x = x.view(-1, self.hidden_size)
        x = self.fc1(x)
        return x
        
class ProgGen_RNN(nn.Module):
    
    def __init__(self, rnn_type, vocab_size, embed_size, hidden_size, num_layers, dropout = 0.5, bidirectional = False):
        super(ProgGen_RNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.embed = torch.nn.Embedding(vocab_size, embed_size)
        
        # RNN Multilayer. Receives (Seq x Batch x Features), Outputs (Seq x Batch x Hidden Size)
        if rnn_type=='lstm':
            self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, dropout = dropout, bidirectional=bidirectional)
        elif rnn_type=='gru':
            self.rnn = nn.GRU(embed_size, hidden_size, num_layers, dropout = dropout, bidirectional=bidirectional)
        else:
            raise NotImplementedError

        # Receives (Seq*Batch x Hidden Size), outputs (Seq*Batch x input_size)
        self.fc1 = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, idxs):
        x = self.embed(idxs)
        x, _ = self.rnn(x)
        x = x.view(-1, self.hidden_size)
        x = self.fc1(x)
        return x