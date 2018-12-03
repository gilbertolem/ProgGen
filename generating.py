from pickle import load, dump
import torch
from numpy import argsort
from sys import argv

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
                if int(thing[0])==4:
                    grouped_structure.append([thing[1:]])
                    grouped_structure.append('|')
                    dur = 0
                else:
                    dur += int(thing[0])
                    if dur>4:
                        grouped_structure.append(compas)
                        grouped_structure.append('|')
                        compas = [thing[1:]]
                        dur = int(thing[0])
                    elif dur==4:
                        compas.append(thing[1:])
                        grouped_structure.append(compas)
                        grouped_structure.append('|')
                        dur = 0
                        compas = []
                    else:
                        compas.append(thing[1:])
        
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

    def compas_to_text(self, compas):
        s = ""
        for chord in compas:
            s += chord + " "
        return s[:-1]
        
def generate_progression(initial_chord = "4C_maj", tune_len = 32, top = 1, model_name = 'model', verbose = False):
    
    # Load model and vocabulary
    model = torch.load('models/'+model_name+'.pt')
    words_num2text = load(open("maps/words_num2text.txt",'rb'))
    words_text2num = load(open("maps/words_text2num.txt",'rb'))
    vocab_size = len(words_text2num)
    model.eval()
    with torch.no_grad():
        
        # Transform initial_chord to tensor (1 x 1 x vocab_size)
        x = torch.zeros(1, 1, vocab_size)
        idx = words_text2num[initial_chord]
        x[0, 0, idx] = 1.0
        predictions = [idx]

        for n in range(tune_len):
            
            logits = model(x)       # input: [1+n x 1 x vocab_size];  output: [(1+n)*1 x vocab_size]
            p = torch.nn.functional.softmax(logits, dim=1)[-1]      # output: [vocab_size]
            p[argsort(p)[:-top]] = 0
            p /= torch.sum(p)
            
            idx = torch.multinomial(p,1).item()
            predictions.append(idx)
            
            new_x = torch.zeros(1, 1, vocab_size)
            new_x[0, 0, idx] = 1.0
            
            x = torch.cat( (x, new_x), 0)
        
    structure = [words_num2text[idx] for idx in predictions]
    if verbose:
        print(structure)
    return Progression(structure)

if __name__=="__main__":

    # Define parameters for generation
    if len(argv) < 4:
        initial_chord = "4C_maj"
        tune_len = 32
        top = 1                 # Choose between the top 'top' probabilities in every iteration
    else:
        initial_chord = argv[1]
        tune_len = int(argv[2])
        top = int(argv[3])
    
    generate_progression(initial_chord, tune_len, top)