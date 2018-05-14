# subtractor

## Goal
- 參考 [IKMLab adder](https://github.com/IKMLab/Adder-practice) 完成 subtractor
- 結合 adder 和 subtractor

## Data generation
```
$ python gen.py
```
- 共 `TRAINING_SIZE` 筆
- 加法減法各一半，避免其中一個 data 數量過少
- adder: A + B (A,B:3 digits)
- subtractor: A – B (A,B:3 digits, A>=B)
- 將隨機 `TRAINING_SIZE / 10` 筆資料作為 validation set

## Implementation

### usage
```
usage: main.py [-h] [-p]

optional arguments:
  -h, --help  show this help message and exit
  -p, --plot  plot the training process
```

### framework
- 利用一個簡單版的 sequence to sequence model （eg. 沒有將 decoder 前一步的 output 當作現在的 input）
- 建構一個 many to many 的 rnn model

![framework](https://github.com/HaoTse/subtractor/blob/master/img/framework.png)

## Experiment

### Size of data and Number of epochs
考慮三位數加法與減法的情況下，總共的組合有 1,500,000 種，首先拿 15,000 筆做為 data 訓練，並使用 1 layer lstm，hidden output size 及 batch size 為 128，在 100 個 epoch 的 accuracy 變化曲線如下圖

![figure 1](https://github.com/HaoTse/subtractor/blob/master/img/Figure_1.png)

可以發現 testing 的準確率可以到9成多，但 validation 卻收斂在7成多而已，推斷為 data size 不足，因此將 data 增加至 30,000 筆試試

![figure 2](https://github.com/HaoTse/subtractor/blob/master/img/Figure_2.png)

由上面兩張圖也可以發現，epoch 數量約在80以上就開始收斂，因此將 epoch 數量定在80，並繼續增加 data 的 size 做實驗

|            | 30,000 | 45,000 | 60,000 | 75,000 | 90,000 | 105,000 |
| ---------- | ------ | ------ | ------ | ------ | ------ | ------- |
| testing    | 0.98   | 0.99   | 0.99   | 0.99   | 0.99   | 0.99    |
| validation | 0.92   | 0.95   | 0.97   | 0.98   | `0.99` | 0.99    |

因此選定 data set 的 size 為 `90,000`，epoch 的數量為 `80`，accuracy 變化曲線如下圖

![figure 3](https://github.com/HaoTse/subtractor/blob/master/img/Figure_3.png)

### Archtiecture of RNN

- Number of layer
    將 RNN 的層數設定為2，accuracy 變化曲線如下圖

    ![figure 4](https://github.com/HaoTse/subtractor/blob/master/img/Figure_4.png)

    可以發現收斂的效果比一層的時候好，因此設定層數為2
- archtiecture
    在一樣的設定下，使用不一樣的架構來做測試

    |          | RNN  | LSTM | GRU  |
    | -------- | ---- | ---- | ---- |
    | training | 0.97 | 0.99 | 0.95 |
    | testing  | 0.95 | 0.99 | 0.94 |
    
    原本認為簡單的加減法不需要使用到 LSTM，但實驗結果顯示 LSTM 表現的確實比單純的 RNN 來的好，也比 GRU 來的好，因此一樣使用 LSTM

### Number of digits

- 1位數 +/- 1位數: data 數量太少，model無法訓練成功，使用一樣的 model，並取100筆資料，結果如下

    ![figure 5](https://github.com/HaoTse/subtractor/blob/master/img/Figure_5.png)
- 2位數 +/- 2位數: data 數量取 10,000 筆，雖能成功收斂，但需要更大 epoch 數量，結果如下

    ![figure 6](https://github.com/HaoTse/subtractor/blob/master/img/Figure_6.png)
- 4位數 +/- 4位數: 使用跟3位數一樣的 data 數量以及設定，結果如下

    ![figure 7](https://github.com/HaoTse/subtractor/blob/master/img/Figure_7.png)

    結果是有成功收斂的，不過加大 data 的數量會再提高準確度

## Reference
- [keras adder example](https://github.com/keras-team/keras/blob/master/examples/addition_rnn.py)