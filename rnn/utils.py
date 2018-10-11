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
from pymongo import MongoClient, DESCENDING, ASCENDING


def get_segment_words_from_api(mpid):
    url = 'http://10.16.57.57:8887/rec/content/' \
          'segmentation/v1?mpId={}'.format(mpid)
    html = requests.get(url).text
    infos = json.loads(html)

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

