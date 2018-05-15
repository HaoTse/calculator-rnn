import numpy as np
from keras.models import load_model

import myhelp

'''
parameters config
'''
config = myhelp.Config()
ctable = myhelp.CharacterTable(config.chars)

def vectorization(data, length):
    '''
    vectorize the data
    '''
    x = np.zeros((len(data), length, len(config.chars)), dtype=np.bool)
    for i, sentence in enumerate(data):
        x[i] = ctable.encode(sentence, length)
    return x

if __name__ == '__main__':
    # argparse
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',
                       default='model/base_model.h5',
                       help='source model path')
    parser.add_argument('--question',
                       default='data/val_x.txt',
                       help='testing input data')
    parser.add_argument('--answer',
                       default='data/val_y.txt',
                       help='testing output data')
    args = parser.parse_args()

    # read datas
    with open(args.question, 'r', encoding='utf-8') as f:
        val_x = np.array(f.read().splitlines())
    with open(args.answer, 'r', encoding='utf-8') as f:
        val_y = np.array(f.read().splitlines())
    val_x = vectorization(val_x, config.MAXLEN)
    val_y = vectorization(val_y, config.DIGITS + 1)

    # print('Validation Data:')
    # print(val_x.shape)
    # print(val_y.shape)

    # load model
    model = load_model(args.source)
    # validation
    score = model.evaluate(val_x, val_y, verbose=0)
    
    print()
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])