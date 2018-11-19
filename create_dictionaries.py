from os import listdir
import xml.etree.ElementTree as ET
import classes
from pickle import dump

# This creates the kind dictionary for mapping from any chord in iRealPro to simple chord functions
def create_simplekind_dictionary():
    simple_kind = {}
    simple_kinds = []
    with open("Dictionaries/simplified_kinds.csv", "rb") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().decode('utf8').split(";")
            kind = line[0]
            sim = line[1]
            simple_kind[kind] = sim
            simple_kinds.append(sim)
    simple_kinds = list(set(simple_kinds))
    dump(simple_kind, open('Dictionaries/simple_kind.txt', 'wb'))

def create_word_dictionary():
    directory = "Tunes/"

    superstructure = []
    for file in listdir(directory):
        for shift in range(12):
            tree = ET.parse(directory + file)
            t = classes.Tune(tree, shift)
            superstructure += t.structure

    different_words = list(set(superstructure))
    words_text2num = {text:num for num, text in enumerate(different_words)}
    words_num2text = {num:text for num, text in enumerate(different_words)}
    dump(words_text2num, open('Dictionaries/words_text2num.txt','wb'))
    dump(words_num2text, open('Dictionaries/words_num2text.txt','wb'))
    
create_simplekind_dictionary()
create_word_dictionary()