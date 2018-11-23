import utils.tools as tools
from utils.nets import ProgGen
from pickle import load, dump
import torch
import matplotlib.pyplot as plt

xml_directory = "XML_Tunes/"

# Load vocabulary
words_text2num = load(open("maps/words_text2num.txt",'rb'))
vocab_size = len(words_text2num)

# Create training data
mode = "all_keys"
filters = {'author':None, 'style':None}
X = tools.musicxml2tensor(xml_directory, words_text2num, mode = mode, filters = filters) # (Seq x Batch x vocab_size)
Y = torch.cat( (X[1:], X[0].unsqueeze(0)) )
Y_target = torch.argmax(Y.view(-1, vocab_size), 1)

# Construct Neural Net
input_size = vocab_size
hidden_size = 512
num_layers = 4
dropout = 0.5
model = ProgGen(input_size, hidden_size, num_layers, dropout)

# Define loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
lr = 1e-2
optim = torch.optim.Adam(model.parameters(), lr=lr)

# Start training
epochs = 100        
train_loss = []
print("\n--------------------------------------------------------------------")
print("TRAINING MODEL...", "\n\nEpoch |", "Training Loss |")
for epoch in range(epochs):
    
    # Forward pass
    logits = model(X)
    loss = loss_fn(logits, Y_target)
    train_loss.append( loss.item() )
    
    # Print to terminal
    print("{:>5} | {:13} |".format(epoch, round(loss.item(),6) ) )
        
    # Backward pass and optimize
    optim.zero_grad()
    loss.backward()
    optim.step()

dump(model, open('models/model.dat','wb'))
plt.plot(train_loss)
plt.show()
