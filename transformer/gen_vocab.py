# -*- coding: utf-8 -*-
'''
This file is used to make a vocabulary from a given corpus.

June.30 2017 by ymthink
yinmiaothink@gmail.com
'''

import codecs
import os
import re
import requests
import json
from collections import Counter
from hyperparams import Hyperparams as hp
from utils import get_recent_mpid_list
from gen_data import *
from multiprocessing import Queue, Pool


def gen_vocab(files, source_file, target_file):
    source_word_cnt = Counter()
    target_word_cnt = Counter()
    for file in files:
        fin = codecs.open(file, 'r', 'utf-8')
        while True:
            text = fin.readline()
            if not text:
                break
            sources, targets = text.split('\t')
            source_word_cnt.update(sources.split())
            target_word_cnt.update(targets.split())

    with codecs.open('{}'.format(source_file), 'w', 'utf-8') as fout:
        fout.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<S>", "</S>"))
        for word, cnt in source_word_cnt.most_common(len(source_word_cnt)):
            if(cnt == 1):
                continue
            fout.write(u"{}\t{}\n".format(word, cnt))

    with codecs.open('{}'.format(target_file), 'w', 'utf-8') as fout:
        fout.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<S>", "</S>"))
        for word, cnt in target_word_cnt.most_common(len(target_word_cnt)):
            if(cnt == 1):
                continue
            fout.write(u"{}\t{}\n".format(word, cnt))


def get_vocab_from_mpid_list(mpid_list):
    word_cnt = Counter()
    data = []

    for mpid in mpid_list:
        url = 'http://10.16.57.57:8887/rec/content/' \
              'segmentation/v1?mpId={}'.format(mpid)
        html = requests.get(url).text
        infos = json.loads(html)

        if infos['status'] == 0:
            print('Invalid mpid: {}'.format(mpid))
            continue

        try:
            title_words = infos['data']['title']
            content_words = infos['data']['content']
        except:
            print(infos)
            continue

        title_list = [_['word'] for _ in title_words if
                          _['posType'] != 'w' and _['posType'] != 'mq' and _['posType'] != 'x']

        content_list = [_['word'] for _ in content_words if
                            _['posType'] != 'w' and _['posType'] != 'mq' and _['posType'] != 'x']
        keywords_list = [_['word'] for _ in title_words if _['posType'].find('n') > -1]

        title = ' '.join(title_list)
        content = ' '.join(content_list)
        keywords = ' '.join(keywords_list)

        if len(title_list) > 0 and len(keywords_list) > 0:
            data.append(u"{}\t{}\n".format(keywords, title))

        content = content.replace('\n', ' ')
        content = content.replace('　', '')
        p = re.compile(' +')
        content = re.sub(p, ' ', content)
        title = re.sub(p, ' ', title)
        content = content.split(' ')
        title = title.split(' ')
        try:
            content.remove('')
        except:
            pass
        try:
            title.remove('')
        except:
            pass

        word_cnt.update(title)
        word_cnt.update(content)
    return {'cnt':word_cnt, 'data':data}

def get_data_from_mpid_list(mpid_list, file_name):
    with codecs.open('{}'.format(file_name), 'w', 'utf-8') as fout:
        for mpid in mpid_list:
            url = 'http://10.16.57.57:8887/rec/content/' \
                  'segmentation/v1?mpId={}'.format(mpid)
            html = requests.get(url).text
            infos = json.loads(html)

            if infos['status'] == 0:
                print('Invalid mpid: {}'.format(mpid))
                continue

            try:
                title_words = infos['data']['title']
            except:
                print(infos)
                continue

            title_list = [_['word'] for _ in title_words if
                                _['posType'] != 'w' and _['posType'] != 'mq' and _['posType'] != 'x']
            title = ' '.join(title_list)

            keywords_list = [_['word'] for _ in title_words if _['posType'].find('n') > -1]
            keywords = ' '.join(keywords_list)

            if len(title_list) > 0 and len(keywords_list) > 0:
                fout.write(u"{}\t{}\n".format(keywords, title))


def write_vocab_file(word_cnt, target_file):

    with codecs.open('{}'.format(target_file), 'w', 'utf-8') as fout:
        fout.write(
            "{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<S>",
                                                                                      "</S>"))
        for word, cnt in word_cnt.most_common(len(word_cnt)):
            if (cnt == 1):
                continue
            fout.write(u"{}\t{}\n".format(word, cnt))

def write_data_file(datas, target_file):
    with codecs.open('{}'.format(target_file), 'w', 'utf-8') as fout:
        for data in datas:
            fout.write(data)


if __name__ == '__main__':
    mpids_list = get_recent_mpid_list(20, 24, channel=10)
    word_cnt = Counter()
    res_list = []
    datas = []
    p = Pool(4)
    for mpid_list in mpids_list:
        res = p.apply_async(get_vocab_from_mpid_list, args=(mpid_list,))
        res_list.append(res)

    p.close()
    p.join()

    for res in res_list:
        item = res.get()
        word_cnt = word_cnt + item['cnt']
        datas.extend(item['data'])

    write_vocab_file(word_cnt, 'vocab.tsv')
    write_data_file(datas, 'train.txt')
    # files = []
    # files.append(hp.train_file)
    # files.append(hp.test_file)
    # gen_vocab(files, 'source_vocab.tsv', 'target_vocab.tsv')








