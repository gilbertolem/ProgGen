#####################################################################################################################
# Imports
#####################################################################################################################

import torch as tt
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F

#####################################################################################################################
# Simple LSTM classifier with only one layer
#####################################################################################################################

class ProgGen(nn.Module):
    
    def __init__(self, num_inputs, num_layers):
        super(ProgGen, self).__init__()
        
        self.lstm = nn.LSTM(num_inputs, num_inputs, num_layers)
        
    def forward(self, x):
        
        n_chords = x.size(0)
        
        y = tt.zeros_like(x)
        
        for i in range(n_chords):
            out, _ = self.lstms[str(i)](x[:(i+1),:,:])
            y[i,:,:] = out[-1,:,:]
            
        y1 = self.softmax(y[:,:,0:12])
        y2 = self.softmax(y[:,:,12:])
        y = tt.cat( (y1, y2), 2)
        
        return y