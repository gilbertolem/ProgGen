import os
import shutil
from pickle import load, dump

import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras import backend as kk
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import LSTM, GRU, Dense, Embedding, Dropout
from tensorflow.python.ops import nn, math_ops

words_text2num = load(open("maps/words_text2num.txt", 'rb'))
vocab_size = len(words_text2num)

# MONITORING METRICS


def top_1_acc(y_true0, y_pred0):
    y_true = tf.reshape(y_true0, [-1])
    y_pred = tf.reshape(y_pred0, [-1, vocab_size])
    return kk.mean(nn.in_top_k(y_pred, math_ops.cast(y_true, 'int32'), 1), axis=-1)


def top_3_acc(y_true0, y_pred0):
    y_true = tf.reshape(y_true0, [-1])
    y_pred = tf.reshape(y_pred0, [-1, vocab_size])
    return kk.mean(nn.in_top_k(y_pred, math_ops.cast(y_true, 'int32'), 3), axis=-1)


def top_5_acc(y_true0, y_pred0):
    y_true = tf.reshape(y_true0, [-1])
    y_pred = tf.reshape(y_pred0, [-1, vocab_size])
    return kk.mean(nn.in_top_k(y_pred, math_ops.cast(y_true, 'int32'), 5), axis=-1)


# MODEL ARCHITECTURE

def build_model(embed_size, rnn_type, hidden_rnn, num_layers, hidden_fc, dropout_fc, batch_size, verbose=False):
    
    rnn_param_dict = {'units': hidden_rnn, 'return_sequences': True,
                      'activation': 'tanh', 'batch_input_shape': [batch_size, None],
                      'recurrent_activation': 'sigmoid', 'recurrent_dropout': 0,
                      'unroll': False, 'use_bias': True}
    if rnn_type == 'gru':
        rnn = GRU
    elif rnn_type == 'lstm':
        rnn = LSTM
    else:
        raise NotImplementedError
    
    model = Sequential([Embedding(vocab_size, embed_size)])
    for _ in range(num_layers):
        model.add(rnn(**rnn_param_dict))
    if hidden_fc > 0:
        model.add(Dense(hidden_fc))
    if dropout_fc > 0:
        model.add(Dropout(dropout_fc))
    model.add(Dense(vocab_size, activation='softmax'))
    
    # Print architecture of the model
    if verbose:
        print("\nMODEL ARCHITECTURE")
        print(model.summary())

    model.backup_build_dict = {'embed_size': embed_size, 'rnn_type': rnn_type, 'hidden_rnn': hidden_rnn,
                               'num_layers': num_layers, 'hidden_fc': hidden_fc, 'dropout_fc': dropout_fc,
                               'batch_size': batch_size}
    return model


# TRAINING PROCEDURE
def train(model, dataset, optimizer, epochs):
    
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
                  metrics=[top_1_acc, top_3_acc, top_5_acc])

    # Checkpoint info
    shutil.rmtree('./training_checkpoints')
    os.mkdir('./training_checkpoints')
    dump(model.backup_build_dict, open('training_checkpoints/build_dict', 'wb'))
    checkpoint_callback = ModelCheckpoint(filepath="./training_checkpoints/ckpt_{epoch}",
                                          save_weights_only=True, period=max(1, epochs//10))
    
    history = model.fit(dataset, epochs=epochs, callbacks=[checkpoint_callback])
    
    return model, history
