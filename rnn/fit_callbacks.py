# -*- coding:utf-8 -*-
# 
# Author: YIN MIAO
# Time: 2018/10/9 10:17

import numpy as np
from keras.callbacks import Callback


def gen_sequence(model, x):
    x_list = x.tolist()
    y_pred = []
    pred_idx = 10
    while pred_idx and len(y_pred) < 20:
        x_arr = np.array(x_list)[np.newaxis]
        pred_vector = model.predict(x_arr)
        pred_idx = np.argmax(pred_vector)

        y_pred.append(pred_idx)
        x_list = x_list[1:]
        x_list.append(pred_idx)

    y_pred = np.array(y_pred)

    return y_pred


def idx_to_words(idx_list, idx_word_dict):

    words = ' '.join(idx_word_dict[idx] for idx in idx_list)
    return words


class PredictForEpoch(Callback):

    def __init__(self, xs, ys, idx_word_dict, file_path):
        self.xs = xs
        self.ys = ys
        self.idx_word_dict = idx_word_dict
        self.file_path = file_path

    def on_epoch_end(self, epoch, epoch_logs):
        with open(self.file_path, 'a+') as f:
            f.write('Epoch {}: \n'.format(epoch))
            f.write('-' * 50 + '\n')
            for x, y in zip(self.xs, self.ys):
                y_pred = gen_sequence(self.model, x)
                y_pred_words = idx_to_words(y_pred, self.idx_word_dict)
                y_real_words = idx_to_words(y, self.idx_word_dict)
                f.write('Actual: ' + repr(y_real_words) + '\n')
                f.write('Predicted: ' + repr(y_pred_words) + '\n')
                f.write('\n')



