from os import listdir, getcwd
from sys import path
path.append(getcwd())

import xml.etree.ElementTree as ET
import utils.classes as classes
from pickle import dump

# This creates the kind dictionary for mapping from any chord in iRealPro to simple chord functions
def create_simplekind_dictionary():
    simple_kind = {}
    simple_kinds = []
    with open("maps/simplified_kinds.csv", "rb") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().decode('utf8').split(";")
            kind = line[0]
            sim = line[1]
            simple_kind[kind] = sim
            simple_kinds.append(sim)
    simple_kinds = list(set(simple_kinds))
    dump(simple_kind, open('maps/simple_kind.txt', 'wb'))

def create_word_dictionary():
    directory = "XML_Tunes/"

    superstructure = []
    for file in listdir(directory):
        for shift in range(12):
            tree = ET.parse(directory + file)
            t = classes.Tune(tree, shift)
            superstructure += t.structure

    different_words = list(set(superstructure))
    different_words = ['']+different_words          # This ensure that index 0 leads to nothing
    words_text2num = {text:num for num, text in enumerate(different_words)}
    words_num2text = {num:text for num, text in enumerate(different_words)}
    dump(words_text2num, open('maps/words_text2num.txt','wb'))
    dump(words_num2text, open('maps/words_num2text.txt','wb'))

if __name__=="__main__":
    
    create_simplekind_dictionary()
    create_word_dictionary()