#####################################################################################################################
# Imports
#####################################################################################################################

# from helper import *
from definitions import *

#####################################################################################################################
# Chord class. Chords are formed by two parts. Both parts have a vector representation of 0's and 1's.
#   -Pitch: refers to the root note
#   -Kind: refers to the color of the chord (minor, sus4, m7b5b9, etc.)
#####################################################################################################################
 
class Chord():
    
    def __init__(self, harmony, note, shift):
        
        # Duration of the chord (whole, half, etc.)
        dur = str(durations[note.find('type').text])
        
        # Get text of the root of the chord (note plus alteration)
        try:
            pitch_alt = harmony.find('root').find('root-alter').text
        except:
            pitch_alt = 0
        pitch_text = harmony.find('root').find('root-step').text + alterations[int(pitch_alt)]
        pitch_text = pitch_num2text[ (pitch_text2num[pitch_text] + shift)%12 ]
        
        # Kind of chord (m, maj7, sus4, etc.)
        kind_text = self.get_kind_text(harmony)
        simple_kind_text = simple_kind[kind_text]
        
        self.text = dur + pitch_text + '_' + simple_kind_text
        
    def get_kind_text(self, harmony):
        kind_text = ''
        try:
            kind_text += harmony.find('kind').attrib['text']
        except:
            kind_text += ''
        for deg in harmony.findall('degree'):
            if deg.find('degree-type').text=='subtract':
                kind_text += '-'
            kind_text += alterations[int(deg.find('degree-alter').text)] + deg.find('degree-value').text
        return kind_text

    def __str__(self):
        return self.text




#####################################################################################################################
# Tune class. Tunes have a key and a mode. Both have vector representations of 0's and 1's.
#   -Key: the root note of the key
#   -Mode: if the key is major or minor
# Also, the Tune class has a list called 'chords', which contains lists which represent measures, and each measure has
# some Chord objects
#####################################################################################################################
 
class Tune():
    
    def __init__(self, tree, shift=0):
        
        self.shift = shift
        self.title = tree.find('movement-title').text
        self.num_measures = len(tree.find('part').findall('measure'))
        self.structure = self.get_structure(tree)
        self.author = self.get_author(tree)
        
    # Get the vector of Chord objects, from an xml, considering repetitions and endings
    def get_structure(self, tree):
        num_measures = len(tree.find('part').findall('measure'))
        structure = []
        for measure in tree.find('part').findall('measure'):
            repeat = self.get_repeat(measure)
            measure_chords = self.get_measure_chords(measure)
            ending = self.get_ending(measure)
            
            if ending is not None:
                structure.append("~"+ending+".- ")
            if repeat == 'forward':
                structure.append(repetitions['forward'])
            structure += measure_chords
            if repeat == 'backward':
                structure.append(repetitions['backward'])
            
        return structure
        
    def get_measure_chords(self, measure):
        chords = []
        for harmony, note in zip(measure.findall('harmony'), measure.findall('note')):
            chords.append(str(Chord(harmony, note, self.shift)))
        return chords
    
    def get_repeat(self, measure):
        for x in measure.findall('barLine'):
            if x.find('repeat') is not None:
                return x.find('repeat').attrib['direction']
        return None
    
    def get_ending(self, measure):
        for x in measure.findall('barLine'):
            if x.find('ending') is not None:
                return x.find('ending').attrib['number']
        return None
    
    def get_author(self, tree):
        for creator in tree.find('identification').findall('creator'):
            if creator.attrib['type']=='composer':
                return creator.text
        return ""
                
    # String representation (readable) of the Tune object
    def __str__(self):
        s = self.title + "\n"
        durs = 0
        for thing in self.structure:
            if thing[0] in ['1','2','4']:
                ch = str(thing)
                dur = int(ch[0])
                durs += dur
                s += str(thing)[1:]
            else:
                s += thing
            
            if durs==4:
                durs = 0
                s += '|'
            else:
                s += ' '

        return s