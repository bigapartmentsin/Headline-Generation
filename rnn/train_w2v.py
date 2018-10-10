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


def train_w2v():
    if not os.path.exists('./data/'):
        os.mkdir('./data/')

    bodies = []
    headlines = []

    for mpid in mpid_list:
        body, headline = get_segment_words_from_api(mpid)
        bodies.append(body)
        headlines.append(headline)

    if os.path.exists('./data/sohunews.mv'):
        w2v = Word2Vec.load('./data/sohunews.mv')
    else:
        w2v = Word2Vec()

    w2v.build_vocab(bodies)
    w2v.train(bodies, total_examples=w2v.corpus_count, epochs=w2v.iter)

    w2v.save('./data/sohunews.mv')


if __name__ == '__main__':
    train_w2v()

