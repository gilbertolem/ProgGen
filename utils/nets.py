import tensorflow as tf

def build_model(vocab_size, embed_size, rnn_type, bidirectional, hidden_rnn, num_layers, dropout_rnn, hidden_fc, dropout_fc, batch_size):
    
    if rnn_type=='gru':
        rnn = tf.keras.layers.GRU(hidden_rnn, return_sequences=True, batch_input_shape=[batch_size, None], recurrent_activation='sigmoid')
    elif rnn_type=='lstm':
        rnn = tf.keras.layers.LSTM(hidden_rnn, return_sequences=True, batch_input_shape=[batch_size, None], recurrent_activation='sigmoid')
    else:
        raise NotImplementedError
    
    if bidirectional:
        rnn_wrapper = tf.keras.layers.Bidirectional(rnn)
    else:
        rnn_wrapper = rnn
    
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embed_size),
    rnn_wrapper,
    tf.keras.layers.Dense(vocab_size, activation='softmax')
    ])
    return model
    
    # # Receives (Seq*Batch x Hidden Size), outputs (Seq*Batch x input_size)
    # if hidden_fc > 0:
    #     self.fc = nn.Sequential(
    #         nn.Linear(self.directions * hidden_rnn, hidden_fc),
    #         nn.Dropout(p=dropout_fc),
    #         nn.Linear(hidden_fc, vocab_size)
    #         )
    # else:
    #     self.fc = nn.Linear(self.directions*hidden_rnn, vocab_size)