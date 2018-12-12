import torch
import pickle

words_text2num = load(open("maps/words_text2num.txt",'rb'))
vocab_size = len(words_text2num)

print(vocab_size)
vector_size = int(vocab_size/2)
embed = torch.nn.Embedding(vocab_size, vector_size)

