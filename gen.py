######### IMPORTS ###########

import tensorflow as tf

from utils.generating import generate_progression

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Disable AVX/FMA warnings

###### GENERATE PROGRESSION #############

dir = '.models/ChickCorea'
initial_chord = "4C_m"
tune_len = 32
top = 5

prog = generate_progression(initial_chord, tune_len, top)
print(prog)