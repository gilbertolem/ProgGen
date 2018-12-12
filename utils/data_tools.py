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
        self.Y = torch.cat( (X[1:], X[0].unsqueeze(0) ) )
    
    def __getitem__(self, index):
        return self.X[:, index], self.Y[:, index]
        
    def __len__(self):
        return self.X.size(1)

def musicxml2tensor(xml_directory, words_text2num, filters = {'author':None, 'style':None}):
    
    """ 
    Function to go through the MusicXML files in xml_directory and convert them to tensors.
    Inputs:
        xml_directory: Name of the folder with the XML files
        words_text2num: Dictionary that maps text to an index
        filters:
            "author": Name of the author to filter
            "style": Style to filter
    Outputs:
        data: pytorch tensor with the dataset in one-hot form
    """
    print("\nCREATING TENSORS FROM MUSICXML FILES...")
    # Read all tunes from the xml_directory and create a list of Tune classes
    tunes = []
    tunes_train = []
    tunes_val = []
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
            val_idx = torch.randperm(12)[:2]
            for shift in range(12):
                tunes.append(classes.Tune(tree, shift))
    
    cut = int(len(tunes)*0.8)
    tunes_train = tunes[:cut]
    tunes_val = tunes[cut:]
    # Shuffle the tunes
    shuffle(tunes_train)
    shuffle(tunes_val)
    
    # Each tune has different length. Final tensor will have the max length of the whole data set
    max_train = max([len(tune) for tune in tunes_train])
    max_val = max([len(tune) for tune in tunes_val])
    max_len = max(max_train, max_val)

    # Create and fill tensor (Sequence x Batch)
    tunes_tensor = torch.zeros(max_len, len(tunes_train)).long()    # All tensor initialized to zero means initialized to blank
    for i, tune in enumerate(tunes_train):
        indexes = torch.tensor(tune.index_form(words_text2num))
        tunes_tensor[0:len(indexes), i] = indexes
    
    print("\t{} tunes succesfully loaded for training.".format(len(tunes_train)))
    
    val_tensor = torch.zeros(max_len, len(tunes_train)).long()    # All tensor initialized to zero means initialized to blank
    for i, tune in enumerate(tunes_train):
        indexes = torch.tensor(tune.index_form(words_text2num))
        val_tensor[0:len(indexes), i] = indexes
    
    print("\t{} tunes succesfully loaded for validation.".format(len(tunes_val)))
    
    return tunes_tensor, val_tensor
    