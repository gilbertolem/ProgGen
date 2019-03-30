import xml.etree.ElementTree as ET
from utils.classes import Tune

path = 'Omnibook/Now\'s_The_Time_1.xml'

tree = ET.parse(path)
tune = Tune(tree, shift=-3)
print(tune)