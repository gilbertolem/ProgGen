from os import listdir
import xml.etree.ElementTree as ET
import utils.classes as classes

xml_directory = 'iReal/'

# Read all tunes from the xml_directory and create a list of Tune classes
authors = []
styles = []
for file in listdir(xml_directory):
    tree = ET.parse(xml_directory + file)
    tune = classes.Progression(tree)

    authors.append(tune.author)
    styles.append(tune.style)
    
from collections import Counter

c_authors = Counter(authors)
c_styles = Counter(styles)

print(c_authors.most_common(30))
print(c_styles.most_common(10))

authors = [a[0] for a in c_authors.most_common(30)]
authors.sort()

styles = [a[0] for a in c_styles.most_common(10)]
styles.sort()

with open('Styles.txt', 'w') as f:
    for item in styles:
        f.write("%s\n" % item)

with open('Authors.txt', 'w') as f:
    for item in authors:
        f.write("%s\n" % item)