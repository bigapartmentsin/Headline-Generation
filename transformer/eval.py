# -*- coding: utf-8 -*-

from __future__ import print_function
import codecs
import os

import tensorflow as tf
import numpy as np

from hyperparams import Hyperparams as hp
from load_data import *
from model import *
from nltk.translate.bleu_score import corpus_bleu
import heapq


def eval(out_file, beam_size=0):
    X, Y, sources, targets = load_data(hp.test_file, hp.maxlen)
    source_word_index, source_index_word = load_vocab(hp.source_vocab_file)
    target_word_index, target_index_word = load_vocab(hp.target_vocab_file)

    # Load graph
    g = Transformer(
        source_vocab_size = len(source_word_index),
        target_vocab_size = len(target_word_index),
        SIGMA = 1e-3,
        LAMBDA = 10,
        is_training = False
    )
    
     
    # Start session         
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        ## Restore parameters
        if tf.train.get_checkpoint_state('./backup/'):
            saver = tf.train.Saver()
            saver.restore(sess, './backup/')
            print('********Restore the latest trained parameters.********')
        else:
            raise Exception('********The model is not existed.********')

        ## Inference
        with codecs.open(out_file, "w", "utf-8") as fout:
            # list_of_refs, hypotheses = [], []
            if beam_size > 0:
                for i in range(len(X)):
                ### Get mini-batches
                    x = X[i: (i+1)]
                    source = sources[i: (i+1)]
                    target = targets[i: (i+1)]
                    preds = np.zeros((beam_size, hp.maxlen), np.int32)
                    scores = np.zeros((beam_size,))

                    # initiative beam search
                    logits = sess.run(g.logits, {g.x: x, g.y: preds[0:1]})
                    indeces = heapq.nlargest(beam_size, range(len(logits[0, 0, :])), logits[0, 0, :].take)
                    # scores += np.log(logits[0, 0][indeces])
                    preds[:, 0] = indeces

                    for j in range(1, hp.maxlen):
                        cur_scores = np.zeros((beam_size * beam_size,))
                        cur_indeces = np.zeros((beam_size * beam_size,))
                        for k in range(beam_size):
                            logits = sess.run(g.logits, {g.x: x, g.y: preds[k:(k+1)]})
                            indeces = heapq.nlargest(beam_size, range(len(logits[0, j, :])), logits[0, j, :].take)
                            # cur_scores[beam_size*k:beam_size*(k+1)] = scores[k] + np.log(logits[0, j][indeces])
                            cur_scores[beam_size*k:beam_size*(k+1)] = logits[0, j][indeces]
                            cur_indeces[beam_size*k:beam_size*(k+1)] = indeces
                        indeces = heapq.nlargest(beam_size, range(len(cur_scores)), cur_scores.take)
                        preds[:, j] = cur_indeces[indeces]
                        # scores = cur_scores[indeces]

                    ### Write to file
                    for k in range(beam_size):
                        s = source[0]
                        t = target[0]
                        got = " ".join(target_index_word[idx] for idx in preds[k]).split("</S>")[0].strip()
                        # fout.write(got +"\t" + s + "\n")
                        fout.write("- keywords: " + s + "\n")
                        fout.write("- title: " + t + "\n")
                        fout.write("- generated: " + got + "\n\n")
                        fout.flush()

                        # bleu score
                        # ref = t.split()
                        # hypothesis = got.split()
                        # if len(ref) > 3 and len(hypothesis) > 3:
                        #    list_of_refs.append([ref])
                        #    hypotheses.append(hypothesis)
            else:
                for i in range(len(X)):
                    x = X[i: (i+1)]
                    source = sources[i: (i+1)]
                    target = targets[i: (i+1)]
                    preds = np.zeros((1, hp.maxlen), np.int32)
                    for j in range(hp.maxlen):
                        _preds = sess.run(g.preds, {g.x: x, g.y: preds})
                        preds[:, j] = _preds[:, j]

                    ### Write to file
                    for s, t, pred in zip(source, target, preds): # sentence-wise
                        got = " ".join(target_index_word[idx] for idx in pred).split("</S>")[0].strip()
                        fout.write("- source: " + s + "\n")
                        fout.write("- expected: " + t + "\n")
                        fout.write("- got: " + got + "\n\n")
                        fout.flush()
          

if __name__ == '__main__':
    eval('./out.txt', beam_size=3)

