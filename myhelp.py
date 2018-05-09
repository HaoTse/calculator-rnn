import numpy as np
from keras import layers

class Config():
    def __init__(self):
        self.TRAINING_SIZE = 40000
        self.DIGITS = 3
        self.REVERSE = False
        self.MAXLEN = self.DIGITS + 1 + self.DIGITS
        self.chars = '0123456789+ '
        self.RNN = layers.LSTM
        self.HIDDEN_SIZE = 128
        self.BATCH_SIZE = 128
        self.LAYERS = 1
        self.EPOCH = 75

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

class CharacterTable(object):
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
    
    def encode(self, C, num_rows):
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x
    
    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return "".join(self.indices_char[i] for i in x)