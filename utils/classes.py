from maps.definitions import durations, alterations, repetitions, pitch_num2text, pitch_text2num
from pickle import load

simple_kind = load(open('maps/simple_kind.txt', 'rb'))


class Measure:

    def __init__(self, measure, shift):

        self.chords = []
        self.notes = []

        for element in measure.getchildren():
            if element.tag == 'harmony':
                self.chords.append(Chord(element, shift=shift))
                self.notes.append([])
            elif element.tag == 'note':
                self.notes[-1].append(Note(element, shift))

    def get_chord_durations(self):
        for chord, notes in zip(self.chords, self.notes):
            dur = 0
            for note in notes:
                dur += note.dur
            chord.dur = dur

    def __str__(self):
        for chord, notes in zip(self.chords, self.notes):
            s = str(chord) + '\t'
            for note in notes:
                s += str(note) + ' '
            s += '\n'
        return s


class Note:

    def __init__(self, element, shift):

        if element.find('pitch') is None:
            self.pitch = "REST"
        else:
            alt = element.find('pitch').find('alter')
            alteration_num = 0 if alt is None else int(alt.text)
            alteration = alterations[alteration_num]
            tentative_pitch = element.find('pitch').find('step').text + str(alteration)
            normalized_pitch = pitch_num2text[(pitch_text2num[tentative_pitch] + shift) % 12]
            self.pitch = normalized_pitch + '_' + str(element.find('pitch').find('octave').text)

        self.dur = element.find('duration').text

    def __str__(self):
        return self.dur + '_' + self.pitch


class Tune:
    
    def __init__(self, tree, shift=0):
        
        self.shift = shift
        self.title = tree.find('work').find('work-title').text
        
        self.structure = [Measure(measure, shift) for measure in tree.find('part').findall('measure')]
        
    def __str__(self):
        s = ''
        for m in self.structure:
            s += str(m) + '\n'
        return s


def get_kind_text(harmony):
    kind = harmony.find('kind').attrib
    kind.setdefault('text', '')
    kind_text = kind['text']
    for deg in harmony.findall('degree'):
        if deg.find('degree-type').text == 'subtract':
            kind_text += '-'
        kind_text += alterations[int(deg.find('degree-alter').text)] + deg.find('degree-value').text
    return kind_text


def get_kind_text_2(harmony):
    kind = harmony.find('kind').text
    if kind == 'major':
        return 'maj'
    elif kind == 'minor':
        return 'm'
    elif kind == 'dominant':
        return '7'
    else:
        print("KIND: {}".format(kind))
        raise NotImplementedError


class Chord:

    def __init__(self, harmony, note=None, shift=0):

        # Get text of the root of the chord (note plus alteration)
        alt = harmony.find('root').find('root-alter')
        pitch_alt = 0 if alt is None else alt.text
        pitch_text = harmony.find('root').find('root-step').text + alterations[int(pitch_alt)]
        self.pitch_text = pitch_num2text[(pitch_text2num[pitch_text] + shift) % 12]

        if note is None:
            self.dur = ''
            # Kind of chord (m, maj7, sus4, etc.)
            self.simple_kind_text = get_kind_text_2(harmony)
        else:
            # Duration of the chord (whole, half, etc.)
            self.dur = str(durations[note.find('type').text])
            # Kind of chord (m, maj7, sus4, etc.)
            kind_text = get_kind_text(harmony)
            self.simple_kind_text = simple_kind[kind_text]

    def __str__(self):
        return self.dur + self.pitch_text + '_' + self.simple_kind_text


def get_repeat(measure):
    for x in measure.findall('barLine'):
        if x.find('repeat') is not None:
            return x.find('repeat').attrib['direction']
    return None


def get_ending(measure):
    for x in measure.findall('barLine'):
        if x.find('ending') is not None:
            return x.find('ending').attrib['number']
    return None


def get_author(tree):
    for creator in tree.find('identification').findall('creator'):
        if creator.attrib['type'] == 'composer':
            return creator.text
    return ""


def get_style(tree):
    for creator in tree.find('identification').findall('creator'):
        if creator.attrib['type'] == 'lyricist':
            return creator.text
    return ""


def get_key(tree):

    for measure in tree.find('part').findall('measure'):

        # The key and mode are always specified in the first measure
        if measure.attrib['number'] == "1":

            # The key pitch is specified in the form of fifths from C. Can be positive or negative
            fifths = int(measure.find('attributes').find('key').find('fifths').text)
            if fifths < 0:
                return (-fifths*5) % 12
            else:
                return (fifths*7) % 12


class Progression:
    
    def __init__(self, tree, shift=0):
        
        self.shift = shift
        self.title = tree.find('movement-title').text
        self.num_measures = len(tree.find('part').findall('measure'))
        self.structure = self.get_structure(tree)
        self.author = get_author(tree)
        self.style = get_style(tree)
        self.key = get_key(tree)
    
    # Get the vector of Chord objects, from an xml, considering repetitions and endings
    def get_structure(self, tree):
        structure = []
        for measure in tree.find('part').findall('measure'):
            repeat = get_repeat(measure)
            measure_chords = self.get_measure_chords(measure)
            ending = get_ending(measure)
            
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

    def matrix_form(self, words_text2num):
        matrix = []
        for s in self.structure:
            one_hot = [0 for _ in range(len(words_text2num))]
            one_hot[words_text2num[s]] = 1
            matrix.append(one_hot)
        return matrix
    
    def index_form(self, words_text2num):
        indexes = []
        for s in self.structure:
            indexes.append(words_text2num[s])
        return indexes
    
    # Length (of the structure)
    def __len__(self):
        return len(self.structure)
    
    # String representation (readable) of the Tune object
    def __str__(self):
        s = self.title + "\n"
        durs = 0
        for thing in self.structure:
            if thing[0] in ['1', '2', '4']:
                ch = str(thing)
                dur = int(ch[0])
                durs += dur
                s += str(thing)[1:]
            else:
                s += thing
            
            if durs == 4:
                durs = 0
                s += '|'
            else:
                s += ' '

        return s


def get_pad(chord, reduce=True):
    dur = int(chord[0])
    if dur >= 1 and reduce:
        dur = int(dur/2)
    L = len(chord)-1

    pad = ''
    N = dur*7 + dur + 1
    for _ in range(N-L):
        pad += ' '
    return pad


def is_chord(ch):
    return len(ch) > 0 and ch[0] in ['1', '2', '4']


class GenProgression:

    def __init__(self, structure):
        self.structure = structure
        self.n_bars = 0
        
        # Group structure in compases (4 beats) and structures (casillas, repeticiones, bars)
        grouped_structure = []
        compas = []
        dur = 0
        for thing in structure:        
            
            if not is_chord(thing):
                grouped_structure.append(thing)
            else:
                dur += int(thing[0])
                if dur > 4 and int(thing[0]) == 4:
                    self.n_bars += 1
                    grouped_structure.append(compas)
                    grouped_structure.append('|')
                    grouped_structure.append([thing])
                    grouped_structure.append('|')
                    compas = []
                    dur = 0
                elif dur > 4 and int(thing[0]) == 2:
                    self.n_bars += 1
                    grouped_structure.append(compas)
                    grouped_structure.append('|')
                    compas = [thing]
                    dur = int(thing[0])
                elif dur == 4:
                    compas.append(thing)
                    self.n_bars += 1
                    grouped_structure.append(compas)
                    grouped_structure.append('|')
                    dur = 0
                    compas = []
                else:
                    compas.append(thing)
        
        # Make string from grouped structure
        s = "\n\nGenerated GenProgression:\n"
        compases = 0
        for thing in grouped_structure:
            if not isinstance(thing, list):
                s += thing
            else:
                s += self.compas_to_text(thing)
                compases += 1
                
            if thing == ":|":
                s += '\n'
                compases = 0
        
        # Add changes of line, and remove _
        bars = 0
        self.string_repr = ''
        for i, ch in enumerate(s):
            if ch != '_':
                self.string_repr += ch
            
            if ch == '|':
                bars += 1
            if bars == 4:
                bars = 0
                self.string_repr += '\n'
        
    def __str__(self):
        return self.string_repr

    def compas_to_text(self, compas, reduce=True):
        s = ""
        for chord in compas:
            if int(chord[0]) == 1 and reduce:
                return self.compas_to_text(compas, False)
            s += chord[1:] + get_pad(chord, reduce)
        return s[:-1]
