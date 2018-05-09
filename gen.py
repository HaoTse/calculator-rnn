import numpy as np

import myhelp

'''
parameters config
'''
config = myhelp.Config()
ctable = myhelp.CharacterTable(config.chars)

if __name__ == '__main__':
    '''
    data generation
    '''
    questions = []
    expected = []
    seen = set()
    print('Generating data...')
    while len(questions) < config.TRAINING_SIZE:
        f = lambda: int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(1, config.DIGITS + 1))))
        a, b = f(), f()
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        q = '{}+{}'.format(a, b)
        query = q + ' ' * (config.MAXLEN - len(q))
        ans = str(a + b)
        ans += ' ' * (config.DIGITS + 1 - len(ans))
        if config.REVERSE:
            query = query[::-1]
        questions.append(query)
        expected.append(ans)
    print('Total addition questions:', len(questions))

    # output to file
    print('Output to files...')
    x = np.array(questions)
    y = np.array(expected)

    indices = np.arange(len(y))
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    # train_test_split
    train_x = x
    train_y = y

    split_at = len(train_x) - len(train_x) // 10
    (x_train, x_val) = train_x[:split_at], train_x[split_at:]
    (y_train, y_val) = train_y[:split_at], train_y[split_at:]

    # print('Training Data:')
    # print(x_train.shape)
    # print(y_train.shape)

    # print('Validation Data:')
    # print(x_val.shape)
    # print(y_val.shape)

    with open('data/train_x.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join("".join(x) for x in x_train))
    with open('data/train_y.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join("".join(x) for x in y_train))
    with open('data/val_x.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join("".join(x) for x in x_val))
    with open('data/val_y.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join("".join(x) for x in y_val))