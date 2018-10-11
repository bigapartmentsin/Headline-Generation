# -*- coding:utf-8 -*-
# 
# Author: miaoyin
# Time: 2018/8/7 10:14

import requests
import json
import time
import math
import numpy
import re

from collections import Counter, namedtuple
from operator import attrgetter

SentenceInfo = namedtuple("SentenceInfo", ("order", "score",))

class Summarizer(object):

    _stop_words = frozenset()

    @property
    def stop_words(self):
        return self._stop_words

    @stop_words.setter
    def stop_words(self, words):
        self._stop_words = frozenset(map(self.normalize_word, words))

    def __init__(self, threshold=0.1, epsilon=0.1):
        self.threshold = threshold
        self.epsilon = epsilon

    def __call__(self, mp_id, sent_count=1):
        url = 'http://10.16.57.57:8887/rec/content/' \
              'segmentation/v1?mpId={}'.format(mp_id)
        html = requests.get(url).text
        infos = json.loads(html)

        title_words = infos['data']['title']
        title = ''
        for title_word in title_words:
            title += title_word['word']
        content_words = infos['data']['content']
        sentences, sentences_words = self.merge_sentence_words(content_words)
        if len(sentences_words) == 0:
            return ''.join([_['word'] for _ in title_words])
        tf_metrics = self._compute_tf(sentences_words)
        idf_metrics = self._compute_idf(sentences_words)

        matrix = self._create_matrix(sentences_words, self.threshold, tf_metrics, idf_metrics)
        scores = self.power_method(matrix, self.epsilon)
        summary = self._get_best_sentences(sentences, scores, sent_count)
        return summary

    def _get_best_sentences(self, sentences, scores, sent_count):
        summary = ''
        infos = (SentenceInfo(i, scores[i]) for i in range(len(scores)))
        infos = sorted(infos, key=attrgetter('score'), reverse=True)
        new_infos = []
        cur_count = 0
        for info in infos:
            new_infos.append(info)
            sentences[info.order] = self._process_sentence(sentences[info.order])
            cur_count += 1
            if cur_count >= sent_count:
                break
        infos = sorted(new_infos, key=attrgetter("order"))
        for info in infos:
            summary += sentences[info.order]

        return summary

    def _process_sentence(self, sentence):
        pattern = re.compile(r'^(不仅如此|不过|而且|虽然|但是|然而|而|例如|比如)，*')
        sentence = re.sub(pattern, '', sentence)
        pattern = re.compile(r'^第*([1-9]|一|二|三|四|五|六|七|八|九|十)([，。\.、]| +)')
        sentence = re.sub(pattern, '', sentence)
        return sentence

    def _compute_tf(self, sentences):
        tf_values = map(Counter, sentences)

        tf_metrics = []
        for sentence in tf_values:
            metrics = {}
            max_tf = self._find_tf_max(sentence)

            for term, tf in sentence.items():
                metrics[term] = tf / max_tf

            tf_metrics.append(metrics)

        return tf_metrics

    @staticmethod
    def _find_tf_max(terms):
        return max(terms.values()) if terms else 1

    @staticmethod
    def _compute_idf(sentences):
        idf_metrics = {}
        sentences_count = len(sentences)

        for sentence in sentences:
            for term in sentence:
                if term not in idf_metrics:
                    n_j = sum(1 for s in sentences if term in s)
                    idf_metrics[term] = math.log(sentences_count / (1 + n_j))

        return idf_metrics

    def _create_matrix(self, sentences, threshold, tf_metrics, idf_metrics):
        """
        Creates matrix of shape |sentences|×|sentences|.
        """
        # create matrix |sentences|×|sentences| filled with zeroes
        sentences_count = len(sentences)
        matrix = numpy.zeros((sentences_count, sentences_count))
        degrees = numpy.zeros((sentences_count, ))

        for row, (sentence1, tf1) in enumerate(zip(sentences, tf_metrics)):
            for col, (sentence2, tf2) in enumerate(zip(sentences, tf_metrics)):
                matrix[row, col] = self.cosine_similarity(sentence1, sentence2, tf1, tf2, idf_metrics)

                if matrix[row, col] > threshold:
                    matrix[row, col] = 1.0
                    degrees[row] += 1
                else:
                     matrix[row, col] = 0

        for row in range(sentences_count):
            for col in range(sentences_count):
                if degrees[row] == 0:
                    degrees[row] = 1

                matrix[row][col] = matrix[row][col] / degrees[row]

        return matrix

    @staticmethod
    def cosine_similarity(sentence1, sentence2, tf1, tf2, idf_metrics):
        """
        We compute idf-modified-cosine(sentence1, sentence2) here.
        It's cosine similarity of these two sentences (vectors) A, B computed as cos(x, y) = A . B / (|A| . |B|)
        Sentences are represented as vector TF*IDF metrics.

        :param sentence1:
            Iterable object where every item represents word of 1st sentence.
        :param sentence2:
            Iterable object where every item represents word of 2nd sentence.
        :type tf1: dict
        :param tf1:
            Term frequencies of words from 1st sentence.
        :type tf2: dict
        :param tf2:
            Term frequencies of words from 2nd sentence
        :type idf_metrics: dict
        :param idf_metrics:
            Inverted document metrics of the sentences. Every sentence is treated as document for this algorithm.
        :rtype: float
        :return:
            Returns -1.0 for opposite similarity, 1.0 for the same sentence and zero for no similarity between sentences.
        """
        unique_words1 = frozenset(sentence1)
        unique_words2 = frozenset(sentence2)
        common_words = unique_words1 & unique_words2

        numerator = 0.0
        for term in common_words:
            numerator += tf1[term]*tf2[term] * idf_metrics[term]**2

        denominator1 = sum((tf1[t]*idf_metrics[t])**2 for t in unique_words1)
        denominator2 = sum((tf2[t]*idf_metrics[t])**2 for t in unique_words2)

        if denominator1 > 0 and denominator2 > 0:
            return numerator / (math.sqrt(denominator1) * math.sqrt(denominator2))
        else:
            return 0.0

    @staticmethod
    def power_method(matrix, epsilon):
        transposed_matrix = matrix.T
        sentences_count = len(matrix)
        try:
            p_vector = numpy.array([1.0 / sentences_count] * sentences_count)
        except ZeroDivisionError as e:
            print(e)
            return []
        lambda_val = 1.0

        while lambda_val > epsilon:
            next_p = numpy.dot(transposed_matrix, p_vector)
            lambda_val = numpy.linalg.norm(numpy.subtract(next_p, p_vector))
            p_vector = next_p

        return p_vector

    def merge_sentence_words(self, words):
        sentences_words = []
        sentence_words = []
        sentence = ''
        sentences = []
        length = len(words)
        for i, word in enumerate(words):
            if word['word'] == '\u3000\u3000':
                continue
            if word['word'] != '\n':
                sentence += word['word']
            if word['posType'] != 'w' \
                    and word['posType'] != 'mq' \
                    and word['posType'] != 'x':
                sentence_words.append(word['word'])
            if word['word'] == '。' \
                    or word['word'] == '！' \
                    or word['word'] == '？' \
                    or word['word'][-1] == '\n':
                if i < (length - 1):
                    if words[i + 1]['word'] == '”':
                        continue
                if len(sentence_words) > 0:
                    sentences_words.append(sentence_words)
                    sentences.append(sentence)
                sentence_words = []
                sentence = ''
            elif word['word'] == '”' and i > 0:
                if words[i-1]['word'] == '。' \
                        or words[i-1]['word'] == '！' \
                        or words[i-1]['word'] == '？' \
                        or words[i-1]['word'][-1] == '\n':
                    if len(sentence_words) > 0:
                        sentences_words.append(sentence_words)
                        sentences.append(sentence)
                    sentence_words = []
                    sentence = ''
        if len(sentence_words) > 0:
            sentences_words.append(sentence_words)
            sentences.append(sentence)

        return sentences, sentences_words


def load_news_from_api(channel_id, num):
    api_url = 'http://dev.mp.sohuno.com/newsInterface/channelNews?' \
              'channelId={}&pageNo=1&pageSize={}&subId=57383&type=1'
    #data_url = 'http://data-api.mp.sohuno.com/v2/news/{}'

    response = requests.get(api_url.format(channel_id, num))
    news_list = json.loads(response.text)
    mp_id_list = [_['id'] for _ in news_list]

    return mp_id_list


if __name__ == '__main__':
    # mp_id_list = []
    # channel_id_list = [8, 10, 12, 13, 15, 17, 18, 19, 23, 24, 25]
    # channel_name_list = ['新闻','军事','文化','历史','财经','体育','汽车','娱乐','时尚','健康','教育']
    # for channel_id in channel_id_list:
    #     mp_id_list.extend(load_news_from_api(channel_id, 10))
    #
    # summarizer = SEOSummarizer(threshold=0.1, epsilon=0.01)
    #
    # #print(summarizer(246254082))
    # with open('summary_results_api.txt', 'w', encoding='utf-8') as f:
    #     for i, channel_id in enumerate(channel_id_list):
    #         mp_id_list = load_news_from_api(channel_id, 10)
    #         f.write('*****************************' + channel_name_list[i] + '*****************************\n')
    #         for mp_id in mp_id_list:
    #             f.write(str(mp_id))
    #             f.write('\n')
    #             summary = summarizer(mp_id)
    #             f.write(summary)
    #             f.write('\n')
    summarizer = Summarizer(threshold=0.1, epsilon=0.01)
    print(summarizer(250194214, 2))

