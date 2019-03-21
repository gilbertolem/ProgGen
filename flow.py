import tensorflow as tf

import utils.data_tools as data_tools
from utils.nets import build_model
from pickle import load
import os
# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from numpy import argmin

xml_directory = "XML_Tunes2/"

words_text2num = load(open("maps/words_text2num.txt",'rb'))
words_num2text = load(open("maps/words_num2text.txt",'rb'))
vocab_size = len(words_text2num)



filter_names = None
filter_fracs = None

filters = {'names':filter_names, 'frac':filter_fracs}
X, W =  data_tools.musicxml2tensor(xml_directory, words_text2num, filters=filters) # X: (batch, sequence), W: (batch,)
print(X)
####### CREATE DATASETS ##########
batch_size = 1

def split_input_target(rola):
    return rola[:-1], rola[1:]

dataset = tf.data.Dataset.from_tensor_slices(X).map(split_input_target)

dataset = dataset.shuffle(100).batch(batch_size, drop_remainder=True)

###### CREATE MODEL #############

embed_size = 100
rnn_type = 'lstm'
bidirectional = False
num_layers = 1
hidden_rnn = 100
dropout_rnn = 0.0

# FC layers parameters
hidden_fc = 0
dropout_fc = 0.0

# Create model and loss function

model = build_model(vocab_size, embed_size, rnn_type, bidirectional, hidden_rnn, num_layers, dropout_rnn, hidden_fc, dropout_fc, batch_size)
print(model.summary())
opt = tf.keras.optimizers.Adam(lr=0.01)
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import nn
from tensorflow.python.ops import math_ops
def metr(y_true0, y_pred0):
    y_true = tf.reshape(y_true0, [-1])
    y_pred = tf.reshape(y_pred0, [-1, vocab_size])
    return K.mean(nn.in_top_k(y_pred, math_ops.cast(y_true, 'int32'), 1), axis=-1)

model.compile(optimizer=opt, 
                loss='sparse_categorical_crossentropy',
                metrics=[metr])

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)

epochs = 100

history = model.fit(dataset.repeat(), epochs=epochs, steps_per_epoch=1, callbacks=[checkpoint_callback])

from utils.generating import generate_progression

initial_chord = "4C_maj"
tune_len = 40
top = 1

prog = generate_progression(initial_chord, tune_len, top, verbose = False)
print("\n\nGenerated Progression:\n")
print(prog)
