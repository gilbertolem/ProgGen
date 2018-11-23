from os import listdir
from pickle import dump
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from random import shuffle

import utils.classes as classes

class TuneData(Dataset):
    
    def __init__(self, X):
        self.X = X
        self.Y = torch.argmax(torch.cat( (X[1:], X[0].unsqueeze(0) ) ), 2)
    
    def __getitem__(self, index):
        return self.X[:, index, :], self.Y[:, index]
        
    def __len__(self):
        return self.X.size(1)


def musicxml2tensor(xml_directory, words_text2num, mode = "original", filters = {'author':None, 'style':None}):
    
    """ 
    Function to go through the MusicXML files in xml_directory and convert them to tensors.
    Inputs:
        xml_directory: Name of the folder with the XML files
        words_text2num: Dictionary that maps text to an index
        mode:
            "original": Load the tunes in their original key according to the XML
            "all_keys": Load the tunes in every possible key (12 times the size of the original)
            "all_in_C": Load the tunes but transpose all of the to be in C
        filters:
            "author": Name of the author to filter
            "style": Style to filter
    Outputs:
        data: pytorch tensor with the dataset in one-hot form
    """
    print("\nCREATING TENSORS FROM MUSICXML FILES...")
    # Read all tunes from the xml_directory and create a list of Tune classes
    tunes = []
    for file in listdir(xml_directory):
        tree = ET.parse(xml_directory + file)
        tune = classes.Tune(tree)
        
        # Determine if tune is to be included in the dataset
        if filters['author'] is not None:
            b_author = filters['author']==tune.author
        else:
            b_author = True
        if filters['style'] is not None:
            b_style = filters['style']==tune.style
        else:
            b_style = True
        
        # If tune to be included, append tune depending on the desired mode
        if b_author and b_style:
            if mode=='original':
                tunes.append(tune)
            elif mode=='all_keys':
                for shift in range(12):
                    tunes.append(classes.Tune(tree, shift))
            elif mode=='all_in_C':
                tunes.append(classes.Tune(tree, -tune.key))
            else:
                raise NotImplementedError
    
    # Shuffle the tunes
    shuffle(tunes)
    
    # Each tune has different length. Final tensor will have the max length of the whole data set
    max_len = max([len(tune) for tune in tunes])

    # Create and fill tensor (Sequence x Batch x Feature)
    tunes_tensor = torch.zeros(max_len, len(tunes), len(words_text2num))
    tunes_tensor[:,:,0] = torch.ones(max_len, len(tunes))       # Initialize all the tensor with first word (blank)
    for i, tune in enumerate(tunes):
        matrix = torch.tensor(tune.matrix_form(words_text2num))
        tunes_tensor[0:len(matrix), i, :] = matrix
    
    print("\t{} tunes succesfully loaded.".format(len(tunes)))
    
    return tunes_tensor
    