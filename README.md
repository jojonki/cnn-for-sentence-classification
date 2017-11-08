# cnn-for-sentence-classification
Convolutional Neural Networks for Sentence Classification, Yoon Kim
https://arxiv.org/abs/1408.5882

## What is this?
Positive/Negative classification of text via CNN and pre-trained embeddings. I reproduced this by PyTorch and Keras.
But Keras version use a fixed filter which only use one word embedded vector for its convolution.

![](https://github.com/jojonki/cnn-for-sentence-classification/blob/images/cnn-class.png?raw=true)

## How to train?
```
$ ./download.sh
$ python train.py
```

## Environment
- python 3.6.1
- PyTorch: 0.2.0_3
- Keras 2.0.8
