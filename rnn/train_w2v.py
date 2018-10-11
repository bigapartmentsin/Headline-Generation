# -*- coding:utf-8 -*-
# 
# Author: YIN MIAO
# Time: 2018/10/10 10:10

import os
import requests
import re
import json
import gensim
from gensim.models import Word2Vec
from utils import *


def train_w2v(mpid_list):
    if not os.path.exists('./data/'):
        os.mkdir('./data/')

    bodies, headlines = get_corpus_from_mpid_list(mpid_list)

    if os.path.exists('./data/sohunews.wv'):
        w2v = Word2Vec.load('./data/sohunews.wv')
        print('The existed word2vec model has been loaded.')
    else:
        w2v = Word2Vec()
        print('A new word2vec model has been loaded.')

    print('{} pieces of news will be trained.'.format(len(mpid_list)))

    w2v.build_vocab(bodies)
    w2v.train(bodies, total_examples=w2v.corpus_count, epochs=w2v.iter)

    w2v.save('./data/sohunews.wv')


if __name__ == '__main__':
    mpid_list = get_recent_mpid_list()
    train_w2v(mpid_list)

