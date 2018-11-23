from pickle import load, dump
import torch
from numpy import argsort
from sys import argv

if __name__=="__main__":

    # Load model and vocabulary
    model = load(open('models/model.dat','rb'))
    words_num2text = load(open("maps/words_num2text.txt",'rb'))
    words_text2num = load(open("maps/words_text2num.txt",'rb'))
    vocab_size = len(words_text2num)

    # Define parameters for generation
    if len(argv) < 4:
        initial_chord = "4C_maj"
        tune_len = 32
        top = 1                 # Choose between the top 'top' probabilities in every iteration
    else:
        initial_chord = argv[1]
        tune_len = int(argv[2])
        top = int(argv[3])

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

    print(structure)    