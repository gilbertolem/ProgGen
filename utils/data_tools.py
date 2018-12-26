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

    # Validate that both frac and names are either None or lists
    if (not isinstance(frac, list) and frac is not None) or (not isinstance(names, list) and names is not None):
        raise Exception('Filters have to be in the form of lists')

    # If necessary, create names and frac list
    if names is None:  # Apply trivial filter
        frac = [1.0]
        names = ["ALL"]
    elif frac is None:  # Names were specified but frac didn't. Apply same frac to all
        frac = [1.0/len(names) for _ in len(names)]
    elif len(frac) != len(names):  # Validate that frac and names are the same length
        raise Exception('Lists of filters and weights have to be the same size')

    # If the sum of the specified fractions is less than one, apply trivial filter to the remaining fraction
    if (1.0 - np.sum(frac)) >= 0.05:
        names.append("ALL")
        frac.append(1.0-np.sum(frac))

    # Define list for instances of each class
    class_count = [0 for _ in range(len(names))]

    # Read all tunes from the xml_directory and create a list of Tune classes
    tunes = []
    tune_classes = []
    for file in listdir(xml_directory):
        tree = ET.parse(xml_directory + file)
        tune = classes.Tune(tree)

        # Get index within the name list
        idx = weight_idx(tune, names)
        if idx == -1:   # Tune not to be considered
            continue
        else:
            class_count[idx] += 12
            for shift in range(12):
                tunes.append(classes.Tune(tree, shift))
                tune_classes.append(idx)

    # Normalize count to compute class frequency
    class_count = np.array(class_count) / np.sum(class_count)

    # Get the weights for the loss function
    tune_weights = [frac[i]/class_count[i] for i in tune_classes]

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
