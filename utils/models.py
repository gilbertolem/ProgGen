from pickle import load
import tensorflow as tf
import os
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import nn
from tensorflow.python.ops import math_ops
from tensorflow.keras.layers import LSTM, GRU, Dense, Embedding, Dropout

words_text2num = load(open("maps/words_text2num.txt",'rb'))
vocab_size = len(words_text2num)

### MONITORING METRICS #######

def top_1_acc(y_true0, y_pred0):
    y_true = tf.reshape(y_true0, [-1])
    y_pred = tf.reshape(y_pred0, [-1, vocab_size])
    return K.mean(nn.in_top_k(y_pred, math_ops.cast(y_true, 'int32'), 1), axis=-1)


def top_3_acc(y_true0, y_pred0):
    y_true = tf.reshape(y_true0, [-1])
    y_pred = tf.reshape(y_pred0, [-1, vocab_size])
    return K.mean(nn.in_top_k(y_pred, math_ops.cast(y_true, 'int32'), 3), axis=-1)


def top_5_acc(y_true0, y_pred0):
    y_true = tf.reshape(y_true0, [-1])
    y_pred = tf.reshape(y_pred0, [-1, vocab_size])
    return K.mean(nn.in_top_k(y_pred, math_ops.cast(y_true, 'int32'), 5), axis=-1)


### MODEL ARCHITECTURE #######

def build_model(embed_size, rnn_type, hidden_rnn, num_layers, dropout_rnn, hidden_fc, dropout_fc, batch_size, return_dict=False, verbose=False):
    
    rnn_param_dict = {'units':hidden_rnn, 'return_sequences':True, 'batch_input_shape':[batch_size, None], 'recurrent_activation':'sigmoid','recurrent_dropout':dropout_rnn}
    if rnn_type=='gru':
        rnn = GRU
    elif rnn_type=='lstm':
        rnn = LSTM
    else:
        raise NotImplementedError
    
    model = tf.keras.Sequential([Embedding(vocab_size, embed_size)])
    for _ in range(num_layers):
        model.add(rnn(**rnn_param_dict))
    if hidden_fc>0:
        model.add(Dense(hidden_fc))
    if dropout_fc>0:
        model.add(Dropout(dropout_fc))
    model.add(Dense(vocab_size, activation='softmax'))
    
    # Print architecture of the model
    if verbose:
        print("\nMODEL ARCHITECTURE")
        print(model.summary())
    
    # Save parameters used for constructing the net
    if return_dict:
        build_dict = {'embed_size':embed_size, 'rnn_type':rnn_type, 'hidden_rnn':hidden_rnn, 'num_layers':num_layers, 'dropout_rnn':dropout_rnn, 'hidden_fc':hidden_fc, 'dropout_fc':dropout_fc, 'batch_size':batch_size}
        
        return model, build_dict
    else:
        return model    

### TRAINING PROCEDURE #######

def train(model, dataset, optimizer, epochs):
    
    model.compile(optimizer=optimizer, 
                    loss='sparse_categorical_crossentropy',
                    metrics=[top_1_acc, top_3_acc, top_5_acc])

    # Checkpoint info
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)
    
    history = model.fit(dataset, epochs=epochs, callbacks=[checkpoint_callback])
    
    return model, history