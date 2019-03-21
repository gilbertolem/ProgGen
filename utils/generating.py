import tensorflow as tf

from pickle import load, dump
from numpy import argsort
from numpy.random import multinomial
from sys import argv
from utils.nets import build_model

def get_pad(chord, reduce = True):
    dur = int(chord[0])
    if dur>=1 and reduce:
        dur = int(dur/2)
    L = len(chord)-1

    pad = ''
    N = dur*7 + dur + 1
    for _ in range(N-L):
        pad += ' '
    return pad

class Progression():

    def __init__(self, structure):
        self.structure = structure
        
    def __str__(self):
        
        structure = self.structure
        
        # Group structure in compases (4 beats) and structures (casillas, repeticiones, bars)
        grouped_structure = []
        compas = []
        dur = 0
        for thing in structure:        
            
            if not self.is_chord(thing):
                grouped_structure.append(thing)
            else:
                dur += int(thing[0])
                if dur>4 and int(thing[0])==4:
                    grouped_structure.append(compas)
                    grouped_structure.append('|')
                    grouped_structure.append([thing])
                    grouped_structure.append('|')
                    compas = []
                    dur = 0
                elif dur>4 and int(thing[0])==2:
                    grouped_structure.append(compas)
                    grouped_structure.append('|')
                    compas = [thing]
                    dur = int(thing[0])
                elif dur==4:
                    compas.append(thing)
                    grouped_structure.append(compas)
                    grouped_structure.append('|')
                    dur = 0
                    compas = []
                else:
                    compas.append(thing)
        
        # Make string from grouped structure
        s = ""
        compases = 0
        new_line = False
        for thing in grouped_structure:
            if not isinstance(thing, list):
                s += thing
            else:
                s += self.compas_to_text(thing)
                compases += 1
                
            if thing==":|":
                s += '\n'
                compases = 0
        
        # Add changes of line, and remove _
        bars = 0
        new_s = ''
        for i, ch in enumerate(s):
            if ch!='_':
                new_s += ch
            
            if ch=='|':
                bars += 1
            if bars==4:
                bars = 0
                new_s += '\n'
        return new_s

    def is_chord(self, ch):
        return len(ch)>0 and ch[0] in ['1','2','4']

    def compas_to_text(self, compas, reduce = True):
        s = ""
        for chord in compas:
            if int(chord[0]) == 1 and reduce==True:
                return self.compas_to_text(compas, False)
            s += chord[1:] + get_pad(chord, reduce)
        return s[:-1]
        
def generate_progression(initial_chord = "4C_maj", tune_len = 32, top = 1, use_gpu = False, verbose = False):
    
    # Load model and vocabulary
    words_num2text = load(open("maps/words_num2text.txt",'rb'))
    words_text2num = load(open("maps/words_text2num.txt",'rb'))
    vocab_size = len(words_text2num)
    
    embed_size = 100
    rnn_type = 'lstm'
    bidirectional = False
    num_layers = 1
    hidden_rnn = 100
    dropout_rnn = 0.0

    # FC layers parameters
    hidden_fc = 0
    dropout_fc = 0.0

    batch_size = 1
    
    # Create model and loss function    
    model = build_model(vocab_size, embed_size, rnn_type, bidirectional, hidden_rnn, num_layers, dropout_rnn, hidden_fc, dropout_fc, batch_size)
    
    model.load_weights(tf.train.latest_checkpoint('./training_checkpoints'))
    model.build(tf.TensorShape([1,None]))
    
    # Transform initial_chord to tensor (1 x 1)
    input_id = words_text2num[initial_chord]
    predictions = [input_id]
    
    for i in range(tune_len):
        model.reset_states()    
        input_eval = tf.expand_dims(predictions, 0)
        
        preds = tf.squeeze(model(input_eval), 0) # Returns (sequential, vocab_size)
        
        probs_top, idx_top = tf.math.top_k(preds[-1])
        
        logits_top = tf.expand_dims(tf.math.log(probs_top), 0)
        pred_id = idx_top[tf.random.categorical(logits_top, 1)[0,0].numpy()].numpy()
        
        predictions.append(pred_id)
        
        
    structure = [words_num2text[idx] for idx in predictions]
    if verbose:
        print(structure)
    return Progression(structure)