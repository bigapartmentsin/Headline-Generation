# -*- coding:utf-8 -*-
# 
# Author: YIN MIAO
# Time: 2018/9/29 15:06

import requests
import keras
import json
import re
import os
import numpy as np
from lxml import etree
import gensim
from gensim.corpora.dictionary import Dictionary
from pyltp import Segmentor
from pyltp import SentenceSplitter


URL = 'http://data-api.mp.sohuno.com/v2/news/{}'
LTP_DATA_DIR = '../../package/ltp_data'  # ltp模型目录的路径


def get_news(mpid):

    response = requests.get(URL.format(mpid))
    news = json.loads(response.text)
    title = news['title']
    root = etree.HTML(news['content'])
    sentences = root.xpath("//text()")
    p = re.compile('。|！|？')
    content = ''
    for sentence in sentences:
        if not re.search(p, sentence):
            continue
        content += sentence
    content = content.replace('　', '')

    return title, content


def get_segment_words(sentence):
    cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
    segmentor = Segmentor()  # 初始化实例
    segmentor.load(cws_model_path)  # 加载模型
    words = segmentor.segment(sentence)  # 分词
    segmentor.release()  # 释放模型
    return words


def get_segment_words_from_api(mpid):
    url = 'http://10.16.57.57:8887/rec/content/' \
          'segmentation/v1?mpId={}'.format(mpid)
    html = requests.get(url).text
    infos = json.loads(html)

    title_words = infos['data']['title']
    content_words = infos['data']['content']
    title = ' '.join([_['word'] for _ in title_words if
                        _['posType'] != 'w' and _['posType'] != 'mq' and _['posType'] != 'x'])
    content = ' '.join([_['word'] for _ in content_words if
                           _['posType'] != 'w' and _['posType'] != 'mq' and _['posType'] != 'x'])
    content = content.replace('\n', ' ')
    content = content.replace('　', '')
    p = re.compile(' +')
    content = re.sub(p, ' ', content)
    title = re.sub(p, ' ', title)
    content = content.split(' ')
    title = title.split(' ')
    content.remove('')
    title.remove('')
    return content, title


if __name__ == '__main__':
    mpid_list = [250194214, 256862684, 256891944, 256890421, 258150669]
    bodies = []
    headlines = []
    for mpid in mpid_list:
        body, headline = get_segment_words_from_api(mpid)
        bodies.append(body)
        headlines.append(headline)
    raw_corpus = bodies

    dictionary = gensim.corpora.Dictionary(raw_corpus)
    corpus_bows = [dictionary.doc2bow(text) for text in raw_corpus]
    w2v = gensim.models.Word2Vec()
    w2v.build_vocab(raw_corpus)
    w2v.train(raw_corpus, total_examples=w2v.corpus_count, epochs=w2v.iter)
    w2v.save('sohunews.wv')

    # ------------------------------------------------------------------------
    # construct words dict
    w2v = gensim.models.Word2Vec.load('sohunews.wv')
    gensim_dict = Dictionary()
    gensim_dict.doc2bow(w2v.wv.vocab.keys(), allow_update=True)
    print(gensim_dict)

    word_idx_dict = {word:(idx+1) for idx, word in gensim_dict.items()}
    idx_word_dict = {(idx+1):word for idx, word in gensim_dict.items()}
    word_idx_dict['\n'] = 0
    idx_word_dict[0] = '\n'

    word_vec_dict = {word:w2v.wv[word] for idx, word in gensim_dict.items()}
    vec_dim = next(len(value) for value in word_vec_dict.values())
    word_vec_dict['\n'] = np.zeros((vec_dim))
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # construct embedding_weights
    n_words = len(word_vec_dict)
    embedding_weights = np.zeros((n_words, vec_dim))
    for word, idx in word_idx_dict.items():
        embedding_weights[idx,:] = word_vec_dict[word]

    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # construct data
    bodies =
    bodies_idx = []
    headlines_idx = []
    for body, headline in zip(bodies, headlines):
        body_idx = []
        headline_idx = []
        for word in body:
            if word in word_idx_dict:
                body_idx.append(word_idx_dict[word])
        for word in headline:
            if word in word_idx_dict:
                headline_idx.append(word_idx_dict[word])
        bodies_idx.append(body_idx)
        headlines_idx.append(headline_idx)

    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # reformat the inputs and outputs
    maxlen = 100
    step = 1

    vocab_size = len(embedding_weights)

    xs = []
    ys = []

    for body, headline in zip(bodies_idx, headlines_idx):
        len_body, len_headline = len(body), len(headline)
        max_headline_len = (len_body - maxlen) // step
        headline.append(0)

        if len_headline <= max_headline_len:
            for idx, word in enumerate(headline):
                x = body[idx:maxlen] + [0] + headline[:idx]
                y = headline[idx]

                xs.append(x)
                ys.append(y)

    ys = keras.utils.np_utils.to_categorical(ys, num_classes=vocab_size)
    xs = np.array(xs, dtype='int32')

    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # build the model
    input = keras.layers.Input(shape=(input_length,), dtype='int32')
    embedding = keras.layers.Embedding(input_dim=vocab_size, output_dim=vec_dim,
                                       weights=[embedding_weights])(input)
    h = keras.layers.recurrent.GRU(units=1024, return_sequences=True, dropout=0.5)(embedding)
    h = keras.layers.recurrent.GRU(units=1024, return_sequences=False, dropout=0.5)(h)
    h = keras.layers.Dense(vocab_size, activation='softmax')(h)

    model = keras.Model(input, h)
    model.compile(loss='cateforical_crossentropy', optimizer='adagrad')

    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # train the model
    callbacks = []
    batch_size = 32
    model.fit(xs, ys, nb_epoch=10, callbacks=callbacks, validation_split=0.0,
              batch_size=batch_size, shuffle=False)

    # ------------------------------------------------------------------------







