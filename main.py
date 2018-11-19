from os import listdir
import xml.etree.ElementTree as ET
import classes


directory = "Tunes/"
tune = "C-Jam Blues"
tree = ET.parse(directory + tune)

t = classes.Tune(tree, 2)
print(t)
print(t.structure)