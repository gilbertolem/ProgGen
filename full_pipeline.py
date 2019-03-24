######### IMPORTS ###########

import tensorflow as tf

from utils.generating import generate_progression
from utils.data_tools import load_data
from utils.models import build_model, train

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Disable AVX/FMA warnings

######### PARAMETERS ###############

# Dataset parameters
batch_size = 128
filter_names = ['Chick Corea']
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
epochs = 300

######## CREATE AND TRAIN MODEL ########

# Load data
xml_directory = "XML_Tunes/"
filters = {'names':filter_names, 'frac':filter_fracs}
dataset =  load_data(xml_directory, filters, batch_size) # X: (batch, sequence), W: (batch,)

# Create model
model = build_model(embed_size, rnn_type, hidden_rnn, num_layers, hidden_fc, dropout_fc, batch_size, verbose=True)

# Train
# optimizer = tf.keras.optimizers.Adam(lr=lr)
# model, history = train(model, dataset, optimizer, epochs)

###### GENERATE PROGRESSION #############

initial_chord = "4C_maj"
tune_len = 32
top = 3

prog = generate_progression(initial_chord, tune_len, top)
print(prog)