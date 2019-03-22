from os import listdir
import tensorflow as tf
import xml.etree.ElementTree as ET
import numpy as np
from pickle import load
import utils.classes as classes

def weight_idx(tune, names):
        if tune.style in names:
            return names.index(tune.style)
        if tune.author in names:
            return names.index(tune.author)
        if "ALL" in names:
            return names.index("ALL")
        return -1

def load_data(xml_directory, filters, batch_size):
    
    tunes, W = musicxml2tunes(xml_directory, filters)
    X = tunes2tensor(tunes)
    print("\t{} tunes successfully loaded for training.".format(len(tunes)))
    
    def split_input_target(rola):
        return rola[:-1], rola[1:]
    
    dataset = tf.data.Dataset.from_tensor_slices(X).map(split_input_target)
    dataset = dataset.shuffle(100).batch(batch_size, drop_remainder=True)
    
    return dataset
    
def tunes2tensor(tunes):
    
    words_text2num = load(open("maps/words_text2num.txt",'rb'))
    
    print("\nCREATING TENSORS FROM MUSICXML FILES...")
    
    # Each tune has different length. Final tensor will have the max length of the whole data set
    max_len = max([len(tune) for tune in tunes])
    
    # Create and fill tensor (Batch x Sequence)
    all_tunes_int = []
    for tune in tunes:
        indexes = tune.index_form(words_text2num)
        # Pad with zeros
        while len(indexes) < max_len:
            indexes.append(0)
        all_tunes_int.append(indexes)
    return tf.convert_to_tensor(all_tunes_int, dtype=tf.int32)

def musicxml2tunes(xml_directory, filters):
    """ 
    Function to go through the MusicXML files in xml_directory and convert them to Tune classes.
    Inputs:
        xml_directory: Name of the folder with the XML files
        filters:
            "names": Author or style to filter
            "frac": Corresponding desired fraction of each author/style
    Outputs:
        data: tensor with the dataset in one-hot form
    """
    print("\nLOADING MUSICXML FILES...")
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
    
    return tunes, tune_weights
    
