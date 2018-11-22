from maps.definitions import *
from pickle import load

simple_kind = load(open('maps/simple_kind.txt','rb'))

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
 
class Tune():
    
    def __init__(self, tree, shift=0):
        
        self.shift = shift
        self.title = tree.find('movement-title').text
        self.num_measures = len(tree.find('part').findall('measure'))
        self.structure = self.get_structure(tree)
        self.author = self.get_author(tree)
        self.style  = self.get_style(tree)
        self.key = self.get_key(tree)
    
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
        
    def get_style(self, tree):
        for creator in tree.find('identification').findall('creator'):
            if creator.attrib['type']=='lyricist':
                return creator.text
        return ""
        
    # Get values of key and mode from xml
    def get_key(self, tree):
        
        for measure in tree.find('part').findall('measure'):        
            
            # The key and mode are always specified in the first measure
            if measure.attrib['number']=="1":
                
                # The key pitch is specified in the form of fifths from C. Can be positive or negative
                fifths = int(measure.find('attributes').find('key').find('fifths').text)
                if fifths < 0:
                    return (-fifths*5)%12
                else:
                    return (fifths*7)%12
    
    def matrix_form(self, words_text2num):
        matrix = []
        for s in self.structure:
            one_hot = [0 for _ in range(len(words_text2num))]
            one_hot[words_text2num[s]] = 1
            matrix.append(one_hot)
        return matrix
    
    # Length (of the structure)
    def __len__(self):
        return len(self.structure)
    
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