import numpy as np
import codecs
import os
import random

from keras import backend as K
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense, Lambda, Permute, Dropout
from keras.layers import Conv2D, MaxPooling1D

def load_data(fpath, label):
    data = []
    with codecs.open(fpath, 'r', 'utf-8', errors='ignore') as f:
        lines = f.readlines()
        for l in lines:
            l = l.rstrip()
            data.append((l.split(' '), label))
    return data

def vectorize(data, sentence_maxlen, w2i):
    vec_data = []
    labels = []
    for d, label in data:
        vec = [w2i[w] for w in d if w in w2i]
        pad_len = max(0, sentence_maxlen - len(vec))
        vec += [0] * pad_len
        vec_data.append(vec)
        
        labels.append(label)
    vec_data = np.array(vec_data)
    labels = np.array(labels)
    return vec_data, labels

def load_glove_weights(glove_dir, embd_dim, vocab_size, word_index):
    embeddings_index = {}
    f = open(os.path.join(glove_dir, 'glove.6B.' + str(embd_dim) + 'd.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index)) 
    embedding_matrix = np.zeros((vocab_size, embd_dim))
    print('embed_matrix.shape', embedding_matrix.shape)
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

pos = load_data('./dataset/rt-polaritydata/rt-polarity.pos', 1)
neg = load_data('./dataset/rt-polaritydata/rt-polarity.neg', 0)
data = pos + neg

sentence_maxlen = max(map(len, (d for d, _ in data)))
print('sentence maxlen', sentence_maxlen)

vocab = []
for d, _ in data:
    for w in d:
        if w not in vocab: vocab.append(w)
vocab = sorted(vocab)
vocab_size = len(vocab)
print('vocab size', len(vocab))
w2i = {w:i for i,w in enumerate(vocab)}

random.shuffle(data)
vecX, vecY = vectorize(data, sentence_maxlen, w2i)
n_data = len(vecX)
split_ind = (int)(n_data * 0.9)
trainX, trainY = vecX[:split_ind], vecY[:split_ind]
testX, testY = vecX[split_ind:], vecY[split_ind:]

embd_dim = 300
glove_embd_w = load_glove_weights('./dataset', embd_dim, vocab_size, w2i)

def Net(vocab_size, embd_size, sentence_maxlen, glove_embd_w):
    sentence = Input((sentence_maxlen,), name='SentenceInput')
    
    # embedding
    embd_layer = Embedding(input_dim=vocab_size, 
                           output_dim=embd_size, 
                           weights=[glove_embd_w], 
                           trainable=False,
                           name='shared_embd')
    embd_sentence = embd_layer(sentence)
    embd_sentence = Permute((2,1))(embd_sentence)
    embd_sentence = Lambda(lambda x: K.expand_dims(x, -1))(embd_sentence)
    
    # cnn
    cnn = Conv2D(1, 
                 kernel_size=(3, sentence_maxlen),
                 activation='relu')(embd_sentence)
    cnn =  Lambda(lambda x: K.sum(x, axis=3))(cnn)
    cnn = MaxPooling1D(3)(cnn)
    cnn = Lambda(lambda x: K.sum(x, axis=2))(cnn)
    out = Dense(1, activation='sigmoid')(cnn)

    model = Model(inputs=sentence, outputs=out, name='sentence_claccification')
    model.compile(optimizer='adagrad', loss='binary_crossentropy', metrics=['accuracy']) 
    return model

model = Net(vocab_size, embd_dim, sentence_maxlen, glove_embd_w)
print(model.summary())

model.fit(trainX, trainY,
            batch_size=32,
            epochs=10,
            validation_data=(testX, testY)
        )
