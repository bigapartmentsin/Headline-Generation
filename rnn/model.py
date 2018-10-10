# -*- coding:utf-8 -*-
# 
# Author: YIN MIAO
# Time: 2018/10/10 11:54

import os
import keras
import numpy as np
from gensim.models import Word2Vec
from gensim.corpora.dictionary import Dictionary
from fit_callbacks import PredictForEpoch


class HeadlineGenerator(object):

    def __init__(self, maxlen, step=1):

        if not os.path.exists('./data/sohunews.wv'):
            raise Exception('word2vec model does not exist!')

        self.w2v = Word2Vec.load('./data/sohunews.mv')
        self.word_idx_dict, self.idx_word_dict, self.word_vec_dict = self._build_dict()
        self.embedding_weights = self._build_embedding_weights()

        self.maxlen = maxlen
        self.step = step
        self.vec_dim = self.embedding_weights.shape[1]

        self.model = self._build_model()
        self.model.compile(loss='categorical_crossentropy', optimizer='adagrad')

    def _build_model(self):
        # build the model
        vocab_size = len(self.embedding_weights)
        input = keras.layers.Input(shape=(self.maxlen,), dtype='int32')
        embedding = keras.layers.Embedding(input_dim=vocab_size, output_dim=vec_dim,
                                           weights=[self.embedding_weights])(input)
        h = keras.layers.recurrent.GRU(units=1024, return_sequences=True, dropout=0.5)(embedding)
        h = keras.layers.recurrent.GRU(units=1024, return_sequences=False, dropout=0.5)(h)
        h = keras.layers.Dense(vocab_size, activation='softmax')(h)

        return keras.Model(input, h)

    def _build_dict(self):

        gensim_dict = Dictionary()
        gensim_dict.doc2bow(self.w2v.wv.vocab.keys(), allow_update=True)

        word_idx_dict = {word: (idx + 1) for idx, word in gensim_dict.items()}
        idx_word_dict = {(idx + 1): word for idx, word in gensim_dict.items()}
        word_idx_dict['\n'] = 0
        idx_word_dict[0] = '\n'

        word_vec_dict = {word: w2v.wv[word] for idx, word in gensim_dict.items()}
        vec_dim = next(len(value) for value in word_vec_dict.values())
        word_vec_dict['\n'] = np.zeros(vec_dim)

        return word_idx_dict, idx_word_dict, word_vec_dict

    def _build_embedding_weights(self):
        n_words = len(self.word_vec_dict)
        embedding_weights = np.zeros((n_words, vec_dim))
        for word, idx in self.word_idx_dict.items():
            embedding_weights[idx, :] = self.word_vec_dict[word]

        return embedding_weights

    def train(self, batch_size=32, file_path='train.txt'):
        bodies_idx = []
        headlines_idx = []
        for body, headline in zip(bodies, headlines):
            body_idx = []
            headline_idx = []
            for word in body:
                if word in self.word_idx_dict:
                    body_idx.append(self.word_idx_dict[word])
            for word in headline:
                if word in self.word_idx_dict:
                    headline_idx.append(self.word_idx_dict[word])
            bodies_idx.append(body_idx)
            headlines_idx.append(headline_idx)

        xs = []
        ys = []

        for body, headline in zip(bodies_idx, headlines_idx):
            len_body, len_headline = len(body), len(headline)
            max_headline_len = (len_body - (self.maxlen-1)) // self.step
            headline.append(0)

            if len_headline <= max_headline_len:
                for idx, word in enumerate(headline):
                    x = body[idx:(self.maxlen-1)] + [0] + headline[:idx]
                    y = headline[idx]

                    xs.append(x)
                    ys.append(y)

        ys = keras.utils.np_utils.to_categorical(ys, num_classes=self.vocab_size)
        xs = np.array(xs, dtype='int32')

        # train the model
        callbacks = []

        x_test = np.zeros((0, xs.shape[1]))
        y_test = []

        idx = 0
        for headline in headlines_idx[:20]:
            len_headline = len(headline)
            x = xs[idx:(idx + 1)]
            y = np.argmax(ys[idx:(idx + len_headline)], axis=1)
            x_test = np.concatenate([x_test, x])
            y_test.append(y)
            idx += len_headline

        predict_per_epoch = PredictForEpoch(x_test, y_test, self.idx_word_dict, file_path)
        callbacks.append(predict_per_epoch)

        self.model.fit(xs, ys, epochs=10, callbacks=callbacks, validation_split=0.0,
                  batch_size=batch_size, shuffle=False)






