import torch
import utils.data_tools as data_tools
from utils.nets import ProgGenRNN, WeightedLoss
from pickle import load, dump
import matplotlib.pyplot as plt

xml_directory = "XML_Tunes/"
torch.manual_seed(999)
use_gpu = torch.cuda.is_available()

# Load vocabulary
words_text2num = load(open("maps/words_text2num.txt",'rb'))
vocab_size = len(words_text2num)

# Create training data
# filter_names = ['Charlie Parker', 'Thelonious Monk']
# filter_fracs = [0.5, 0.5]
filter_names = None
filter_fracs = None
filters = {'names': filter_names, 'frac': filter_fracs}
Train, Val = data_tools.musicxml2tensor(xml_directory, words_text2num, filters = filters)
train_data = data_tools.TuneData(Train)
val_data = data_tools.TuneData(Val)

# Construct Neural Net
embed_size = 100
hidden_size = 256
num_layers = 1
dropout = 0
bidirectional = True
rnn_type = 'lstm'
model = ProgGenRNN(rnn_type, vocab_size, embed_size, hidden_size, num_layers, dropout, bidirectional)
loss_fn = WeightedLoss()

# Define loader
sampler = torch.utils.data.RandomSampler(train_data)
train_loader = torch.utils.data.DataLoader(train_data, batch_size = 372, sampler = sampler, num_workers = 1 if use_gpu else 4)
val_loader = torch.utils.data.DataLoader(val_data, batch_size = 372, num_workers = 1 if use_gpu else 4)

if use_gpu:
    model = model.cuda()
    loss_fn = loss_fn.cuda()

# Define loss function and optimizer
lr = 1e-2
optim = torch.optim.Adam(model.parameters(), lr=lr)

from utils.training import train
epochs = 5
losses = train(epochs, model, optim, train_loader, val_loader, loss_fn, use_gpu)

plt.semilogy(losses[0], label='Train')
plt.semilogy(losses[1], label='Val')
plt.legend()
plt.show()
import numpy as np
print(losses[0][-1])
print(losses[1][-1])
idx = np.argmin(losses[1])
print(losses[0][idx])
print(losses[1][idx])

from utils.generating import generate_progression

model_name = "model"
initial_chord = "4C_maj"
tune_len = 32
top = 10

prog = generate_progression(initial_chord, tune_len, top, model_name, verbose=False)
print("Generated Progression:\n")
print(prog)