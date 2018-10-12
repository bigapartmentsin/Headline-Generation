# -*- coding:utf-8 -*-
# 
# Author: YIN MIAO
# Time: 2018/10/10 10:37

import requests
import json
import re
import numpy as np
import os, time, random
import requests
import datetime
from lxml import etree
from summarizer import Summarizer
import jieba.analyse
import jieba.posseg
# from pyltp import Segmentor
# from pyltp import Postagger
from pymongo import MongoClient, DESCENDING, ASCENDING


URL = 'http://data-api.mp.sohuno.com/v2/news/{}'
LTP_DATA_DIR = '../../package/ltp_data'  # ltp模型目录的路径


def get_summary(mpid):
    summarizer = Summarizer(0.1, 0.01)
    summary = summarizer(mpid)
    return summary


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


def get_postags(words):
    pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
    postagger = Postagger()
    postagger.load(pos_model_path)

    postags = postagger.postag(words)
    postagger.release()
    return postags


def get_dependencies(words, postags):

    par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`

    parser = Parser() # 初始化实例
    parser.load(par_model_path)  # 加载模型

    arcs = parser.parse(words, postags)  # 句法分析

    print(' '.join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
    parser.release()  # 释放模型
    return arcs


def get_segment_words_from_api(mpid):
    url = 'http://10.16.57.57:8887/rec/content/' \
          'segmentation/v1?mpId={}'.format(mpid)
    html = requests.get(url).text
    infos = json.loads(html)

    if infos['status'] == 0:
        print(mpid)
        print(infos)
        return

    try:
        title_words = infos['data']['title']
        content_words = infos['data']['content']
    except:
        print(infos)
        return
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
    try:
        content.remove('')
    except:
        pass
    try:
        title.remove('')
    except:
        pass
    return content, title


def get_corpus_from_mpid_list(mpid_list):
    bodies = []
    headlines = []

    for mpid in mpid_list:
        body, headline = get_segment_words_from_api(mpid)
        if not body:
            continue
        if not headline:
            continue
        bodies.append(body)
        headlines.append(headline)

    return bodies, headlines


def get_recent_mpid_list(hours=24):
    mpid_list = []

    KEYIMAGE_MONGO_URL = 'mongodb://captain_prod_rw:98A419Z27K9PoS4@captain-prod01.db2.sohuno.com:10000,captain-prod02.db2.sohuno.com:10000/captain_prod?readPreference=secondaryPreferred'
    mpRecNewsLightPart = MongoClient(KEYIMAGE_MONGO_URL).get_database('captain_prod').get_collection(
        'mpRecNewsLightPart')

    end = datetime.datetime.now() - datetime.timedelta(hours=12)
    start = end - datetime.timedelta(hours=hours)

    cursor = mpRecNewsLightPart.find({'postTime': {'$gt': start, '$lt': end}}).sort("createTime", ASCENDING)
    print('From {} to {}, {} pieces of news have been found.'.format(start, end, cursor.count()))
    for i in range(cursor.count()):
        mpid_list.append(cursor.next()['mpId'])

    return mpid_list


def get_keywords_from_mpid_list(mpid_list):
    keywords_list = []
    title_list = []
    for mpid in mpid_list:
        url = 'http://10.16.57.57:8887/rec/content/' \
              'segmentation/v1?mpId={}'.format(mpid)
        html = requests.get(url).text
        infos = json.loads(html)

        try:
            title_words = infos['data']['title']
        except:
            print(infos)
            return
        title = [_['word'] for _ in title_words if
                            _['posType'] != 'w' and _['posType'] != 'mq' and _['posType'] != 'x']

        keywords = [_['word'] for _ in title_words if _['posType'].find('n') > -1]

        keywords_list.append(keywords)
        title_list.append(title)

    return keywords_list, title_list


