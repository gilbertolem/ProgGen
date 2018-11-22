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
    
    def __init__(self, input_size, hidden_size, num_layers, dropout = 0.5):
        super(ProgGen, self).__init__()
        self.hidden_size = hidden_size
        # LSTM Multilayer. Receives (Seq x Batch x Features), Outputs (Seq x Batch x Hidden Size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout = dropout)

        # Receives (Seq*Batch x Hidden Size), outputs (Seq*Batch x input_size)
        self.fc1 = nn.Linear(hidden_size, input_size)
        
    def forward(self, x):
        
        x, _ = self.lstm(x)
        x = x.view(-1, self.hidden_size)
        x = self.fc1(x)
        
        return x