import numpy as np
from keras.models import Sequential
from keras import layers
import matplotlib.pyplot as plt

import myhelp

'''
parameters config
'''
config = myhelp.Config()
colors = myhelp.colors
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
    parser.add_argument('--dest',
                       default='model/base_model.h5',
                       help='output model path')
    parser.add_argument('-p', '--plot',
                       help='plot the training process',
                       action='store_true')
    parser.add_argument('-v', '--valid',
                       help='validation or not',
                       action='store_true')
    args = parser.parse_args()

    # read datas
    with open('data/train_x.txt', 'r', encoding='utf-8') as f:
        train_x = np.array(f.read().splitlines())
    with open('data/train_y.txt', 'r', encoding='utf-8') as f:
        train_y = np.array(f.read().splitlines())
    with open('data/val_x.txt', 'r', encoding='utf-8') as f:
        val_x = np.array(f.read().splitlines())
    with open('data/val_y.txt', 'r', encoding='utf-8') as f:
        val_y = np.array(f.read().splitlines())

    print('Vectorization...')
    train_x = vectorization(train_x, config.MAXLEN)
    train_y = vectorization(train_y, config.DIGITS + 1)
    val_x = vectorization(val_x, config.MAXLEN)
    val_y = vectorization(val_y, config.DIGITS + 1)

    # print('Training Data:')
    # print(train_x.shape)
    # print(train_y.shape)

    # print('Validation Data:')
    # print(val_x.shape)
    # print(val_y.shape)

    # build model
    print('Build model...')
    model = Sequential()
    model.add(config.RNN(config.HIDDEN_SIZE, input_shape=(config.MAXLEN, len(config.chars))))
    model.add(layers.RepeatVector(config.DIGITS + 1))
    for _ in range(config.LAYERS):
        model.add(config.RNN(config.HIDDEN_SIZE, return_sequences=True))

    model.add(layers.TimeDistributed(layers.Dense(len(config.chars))))
    model.add(layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    # model.summary()

    # training
    print()
    print('Training...')
    print('-' * 50)
    history = model.fit(train_x, train_y,
                        batch_size=config.BATCH_SIZE,
                        epochs=config.EPOCH,
                        validation_data=(val_x, val_y))

    # output train history
    if args.plot:
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    # validation
    if args.valid:
        for i in range(10):
            ind = np.random.randint(0, len(val_x))
            rowx, rowy = val_x[np.array([ind])], val_y[np.array([ind])]
            preds = model.predict_classes(rowx, verbose=0)
            tmp = model.predict(rowx)
            q = ctable.decode(rowx[0])
            correct = ctable.decode(rowy[0])
            guess = ctable.decode(preds[0], calc_argmax=False)
            print('Q', q[::-1] if config.REVERSE else q, end=' ')
            print('T', correct, end=' ')
            if correct == guess:
                print(colors.ok + '☑' + colors.close, end=' ')
            else:
                print(colors.fail + '☒' + colors.close, end=' ')
            print(guess)

    # save model
    model.save(args.dest)