from os import listdir
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import numpy as np

import utils.classes as classes


class TuneData(Dataset):
    
    def __init__(self, Data):
        X, W = Data
        self.X = X
        self.Y = torch.cat( (X[1:], X[0].unsqueeze(0) ) )
        self.W = W
    
    def __getitem__(self, index):
        X = self.X[:,index]
        W = self.W[index]*torch.ones_like(X)
        return X, self.Y[:, index], W
        
    def __len__(self):
        return self.X.size(1)


def weight_idx(tune, names):
        if tune.style in names:
            return names.index(tune.style)
        if tune.author in names:
            return names.index(tune.author)
        if "ALL" in names:
            return names.index("ALL")
        return -1


def musicxml2tensor(xml_directory, words_text2num, filters):
    
    """ 
    Function to go through the MusicXML files in xml_directory and convert them to tensors.
    Inputs:
        xml_directory: Name of the folder with the XML files
        words_text2num: Dictionary that maps text to an index
        filters:
            "names": Author or style to filter
            "frac": Corresponding desired fraction of each author/style
    Outputs:
        data: pytorch tensor with the dataset in one-hot form
    """

    print("\nCREATING TENSORS FROM MUSICXML FILES...")

    frac = filters['frac']
    names = filters['names']
    if (not isinstance(frac, list) and frac is not None) or (not isinstance(names, list) and names is not None):
        raise Exception('Filters have to be in the form of lists')

    if names is None:
        b_filter = False
    elif frac is None and names is not None:
        weights = [1.0/len(names) for _ in range(len(names))]
        b_filter = True
    elif (names is not None) and (len(frac) != len(names)):
        raise Exception('Lists of filters and weights have to be the same size')
    else:
        b_filter = True
        weights = [1.0/i for i in frac]

    if b_filter and (1.0 - np.sum(frac)) >= 0.05:
        w = 1.0-np.sum(frac)
        weights.append(w)
        names.append("ALL")

    # Read all tunes from the xml_directory and create a list of Tune classes
    tunes = []
    tune_weights = []
    for file in listdir(xml_directory):
        tree = ET.parse(xml_directory + file)
        tune = classes.Tune(tree)

        if b_filter:
            idx = weight_idx(tune, names)
            if idx == -1:
                continue
            else:
                w = weights[idx]
                for shift in range(12):
                    tunes.append(classes.Tune(tree, shift))
                    tune_weights.append(w)
        else:
            for shift in range(12):
                tunes.append(classes.Tune(tree, shift))
                tune_weights.append(1.0)

    # Split in Training and Validation Set
    cut = int(len(tunes)*0.8)
    tunes_train = tunes[:cut]
    W_train = tune_weights[:cut]
    tunes_val = tunes[cut:]
    W_val = tune_weights[cut:]

    # Shuffle the tunes
    idxs_train = torch.randperm(len(tunes_train))
    tunes_train = [tunes_train[int(i.item())] for i in idxs_train]
    W_train = [W_train[int(i.item())] for i in idxs_train]
    idxs_val = torch.randperm(len(tunes_val))
    tunes_val = [tunes_val[int(i.item())] for i in idxs_val]
    W_val = [W_val[int(i.item())] for i in idxs_val]

    # Each tune has different length. Final tensor will have the max length of the whole data set
    max_train = max([len(tune) for tune in tunes_train])
    max_val = max([len(tune) for tune in tunes_val])
    max_len = max(max_train, max_val)

    # Create and fill tensor (Sequence x Batch)
    X_train = torch.zeros(max_len, len(tunes_train)).long()    # All tensor initialized to zero means initialized to blank
    for i, tune in enumerate(tunes_train):
        indexes = torch.Tensor(tune.index_form(words_text2num))
        X_train[0:len(indexes), i] = indexes
    print("\t{} tunes successfully loaded for training.".format(len(tunes_train)))
    
    X_val = torch.zeros(max_len, len(tunes_val)).long()    # All tensor initialized to zero means initialized to blank
    for i, tune in enumerate(tunes_val):
        indexes = torch.Tensor(tune.index_form(words_text2num))
        X_val[0:len(indexes), i] = indexes
    print("\t{} tunes successfully loaded for validation.".format(len(tunes_val)))
    
    return (X_train, W_train), (X_val, W_val)
