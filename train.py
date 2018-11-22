import utils.tools as tools
from pickle import load

xml_directory = "XML_Tunes/"

words_num2text = load(open("maps/words_num2text.txt",'rb'))
words_text2num = load(open("maps/words_text2num.txt",'rb'))

vocab_size = len(words_num2text)

mode = "original"

X = tools.musicxml2tensor(xml_directory, words_text2num, mode = mode)
