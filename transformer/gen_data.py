# -*- coding:utf-8 -*-
# 
# Author: YIN MIAO
# Time: 2018/10/15 16:22

import codecs
import json
import requests


def gen_data_from_mpid_list(mpid_list, file_name):
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


