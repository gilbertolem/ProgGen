######### IMPORTS ###########

import tensorflow as tf

from utils.generating import generate_progression
from utils.data_tools import load_data
from utils.models import build_model, train

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Disable AVX/FMA warnings

######### PARAMETERS ###############

# Dataset parameters
batch_size = 1
filter_names = ['John Klenner']
filter_fracs = [1.0]

# Model parameters
embed_size = 100
rnn_type = 'lstm'
num_layers = 1
hidden_rnn = 100
dropout_rnn = 0.0
hidden_fc = 0
dropout_fc = 0.0

# Training parameters
lr = 1e-2
epochs = 30

######## CREATE AND TRAIN MODEL ########

# Load data
xml_directory = "XML_Tunes/"
filters = {'names':filter_names, 'frac':filter_fracs}
dataset =  load_data(xml_directory, filters, batch_size) # X: (batch, sequence), W: (batch,)

# Create model
model, build_dict = build_model(embed_size, rnn_type, hidden_rnn, num_layers, dropout_rnn, hidden_fc, dropout_fc, batch_size, return_dict=True, verbose=True)

# Train
optimizer = tf.keras.optimizers.Adam(lr=lr)
model, history = train(model, dataset, optimizer, epochs)

###### GENERATE PROGRESSION #############

initial_chord = "4C_maj"
tune_len = 40
top = 10

prog = generate_progression(build_dict, initial_chord, tune_len, top)
print(prog)