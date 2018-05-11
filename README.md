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
- 建構一個 many to many 的 rnn ｍodel

![framework](https://github.com/HaoTse/subtractor/blob/master/img/framework.png)

## Reference
- [IKMLab adder](https://github.com/IKMLab/Adder-practice)