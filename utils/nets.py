import torch
import torch.nn as nn
import torch.nn.functional as F

#####################################################################################################################
# Weighted CrossEntropy Loss, for combining authors and styles
#####################################################################################################################


class WeightedLoss(nn.Module):
    
    def __init__(self):
        super(WeightedLoss, self).__init__()
        
    def forward(self, x, target, weights):
        element_loss = F.cross_entropy(x, target, reduction='none')
        return (weights.float()*element_loss).mean()

#####################################################################################################################
# RNN generator, with embedding
#####################################################################################################################


class ProgGenRNN(nn.Module):
    
    def __init__(self, rnn_type, vocab_size, embed_size, hidden_size, num_layers, dropout = 0.5, bidirectional = False):
        super(ProgGenRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.directions = 2 if bidirectional else 1
        self.embed = torch.nn.Embedding(vocab_size, embed_size)
        
        # RNN Multilayer. Receives (Seq x Batch x Features), Outputs (Seq x Batch x Hidden Size)
        if rnn_type=='lstm':
            self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, dropout = dropout, bidirectional=bidirectional)
        elif rnn_type=='gru':
            self.rnn = nn.GRU(embed_size, hidden_size, num_layers, dropout = dropout, bidirectional=bidirectional)
        else:
            raise NotImplementedError

        # Receives (Seq*Batch x Hidden Size), outputs (Seq*Batch x input_size)
        self.fc1 = nn.Linear(self.directions*hidden_size, vocab_size)
        
    def forward(self, idxs):
        x = self.embed(idxs)
        x, _ = self.rnn(x)
        x = x.view(-1, self.directions*self.hidden_size)
        x = self.fc1(x)
        return x
