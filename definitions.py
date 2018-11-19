from pickle import dump, load

###################### Pitch definitions ######################

# Durations
durations = {}
durations['whole'] = 4
durations['half'] = 2
durations['quarter'] = 1

# Alterations
alterations = {}
alterations[0] = ''
alterations[-1] = 'b'
alterations[1] = '#'

# Repetitions
repetitions = {}
repetitions['forward'] = '|:'
repetitions['backward'] = ':|'

# Map from text to number
pitch_text2num = {}
pitch_text2num['C'] = 0
pitch_text2num['C#'] = 1
pitch_text2num['Db'] = 1
pitch_text2num['D'] = 2
pitch_text2num['D#'] = 3
pitch_text2num['Eb'] = 3
pitch_text2num['E'] = 4
pitch_text2num['E#'] = 5
pitch_text2num['Fb'] = 4
pitch_text2num['F'] = 5
pitch_text2num['F#'] = 6
pitch_text2num['Gb'] = 6
pitch_text2num['G'] = 7
pitch_text2num['G#'] = 8
pitch_text2num['Ab'] = 8
pitch_text2num['A'] = 9
pitch_text2num['A#'] = 10
pitch_text2num['Bb'] = 10
pitch_text2num['B'] = 11
pitch_text2num['B#'] = 0
pitch_text2num['Cb'] = 11

# Map from number to text
pitch_num2text = {}
pitch_num2text[0] = 'C'
pitch_num2text[1] = 'C#'
pitch_num2text[2] = 'D'
pitch_num2text[3] = 'Eb'
pitch_num2text[4] = 'E'
pitch_num2text[5] = 'F'
pitch_num2text[6] = 'F#'
pitch_num2text[7] = 'G'
pitch_num2text[8] = 'Ab'
pitch_num2text[9] = 'A'
pitch_num2text[10] = 'Bb'
pitch_num2text[11] = 'B'

# Load dictionaries 
simple_kind = load( open('Dictionaries/simple_kind.txt', 'rb'))
words_text2num = load(open('Dictionaries/words_text2num.txt','rb'))
words_num2text = load(open('Dictionaries/words_num2text.txt','rb'))