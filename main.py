import numpy as np
from keras.models import Sequential
from keras import layers

import myhelp

'''
parameters config
'''
config = myhelp.Config()
colors = myhelp.colors
ctable = myhelp.CharacterTable(config.chars)

def vectorization(data, length):
    x = np.zeros((len(data), length, len(config.chars)), dtype=np.bool)
    for i, sentence in enumerate(data):
        x[i] = ctable.encode(sentence, length)
    return x

# read datas
with open('data/train_x.txt', 'r', encoding='utf-8') as f:
    train_x = np.array(f.read().splitlines())
with open('data/train_y.txt', 'r', encoding='utf-8') as f:
    train_y = np.array(f.read().splitlines())
with open('data/val_x.txt', 'r', encoding='utf-8') as f:
    val_x = np.array(f.read().splitlines())
with open('data/val_y.txt', 'r', encoding='utf-8') as f:
    val_y = np.array(f.read().splitlines())
with open('data/test_x.txt', 'r', encoding='utf-8') as f:
    test_x = np.array(f.read().splitlines())
with open('data/test_y.txt', 'r', encoding='utf-8') as f:
    test_y = np.array(f.read().splitlines())

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

# print('Testing Data:')
# print(test_x.shape)
# print(test_y.shape)

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
for iteration in range(1):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(train_x, train_y,
              batch_size=config.BATCH_SIZE,
              epochs=config.EPOCH,
              validation_data=(val_x, val_y))
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

# testing
print('MSG: testing')
test_x = vectorization(test_x, config.MAXLEN)
test_y = vectorization(test_y, config.DIGITS + 1)

right = 0
preds = model.predict_classes(test_x, verbose=0)
for i in range(len(preds)):
    q = ctable.decode(test_x[i])
    correct = ctable.decode(test_y[i])
    guess = ctable.decode(preds[i], calc_argmax=False)
    print('Q', q[::-1] if config.REVERSE else q, end=' ')
    print('T', correct, end=' ')
    if correct == guess:
        print(colors.ok + '☑' + colors.close, end=' ')
        right += 1
    else:
        print(colors.fail + '☒' + colors.close, end=' ')
    print(guess)
print("MSG : Accuracy is {}".format(right / len(preds)))