# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import sys
sys.path.append('./')

from hyperparams import Hyperparams as hp
from load_data import *
from modules import *


class Transformer(object):
    def __init__(self, source_vocab_size, target_vocab_size, SIGMA, LAMBDA, is_training):
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.SIGMA = SIGMA
        self.LAMBDA = LAMBDA
        self.is_training = is_training

        if self.is_training:
            X, Y, _, _ = load_data(hp.train_file, hp.maxlen)

            # calc total batch count
            self.num_batch = len(X) // hp.batch_size
            
            # Convert to tensor
            X = tf.convert_to_tensor(X, tf.int32)
            Y = tf.convert_to_tensor(Y, tf.int32)
            
            # Create Queues
            input_queues = tf.train.slice_input_producer([X, Y])
                    
            # create batch queues
            self.x, self.y = tf.train.shuffle_batch(input_queues,
                                        num_threads=8,
                                        batch_size=hp.batch_size, 
                                        capacity=hp.batch_size*64,   
                                        min_after_dequeue=hp.batch_size*32, 
                                        allow_smaller_final_batch=False)
        else: # inference
            self.x = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
            self.y = tf.placeholder(tf.int32, shape=(None, hp.maxlen))

        # define decoder inputs
        decoder_inputs = tf.concat((tf.ones_like(self.y[:, :1]) * 2, self.y[:, :-1]), -1)  # 2:<S>

        # Encoder
        with tf.variable_scope("encoder"):
            ## Embedding
            enc = embedding(self.x,
                            vocab_size=self.source_vocab_size,
                            num_units=hp.hidden_units,
                            scale=True,
                            scope="enc_embed")

            ## Positional Encoding
            enc += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                             vocab_size=hp.maxlen,
                             num_units=hp.hidden_units,
                             zero_pad=False,
                             scale=False,
                             scope="enc_pe")

            ## Dropout
            enc = tf.layers.dropout(enc,
                                    rate=hp.dropout_rate,
                                    training=tf.convert_to_tensor(self.is_training))

            ## Blocks
            for i in range(hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              num_units=hp.hidden_units,
                                              num_heads=hp.num_heads,
                                              dropout_rate=hp.dropout_rate,
                                              is_training=self.is_training,
                                              causality=False)

                    ### Feed Forward
                    enc = feedforward(enc, num_units=[4 * hp.hidden_units, hp.hidden_units])

        # Decoder
        with tf.variable_scope("decoder"):
            ## Embedding
            dec = embedding(decoder_inputs,
                            vocab_size=self.target_vocab_size,
                            num_units=hp.hidden_units,
                            scale=True,
                            scope="dec_embed")

            ## Positional Encoding
            dec += embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(decoder_inputs)[1]), 0), [tf.shape(decoder_inputs)[0], 1]),
                vocab_size=hp.maxlen,
                num_units=hp.hidden_units,
                zero_pad=False,
                scale=False,
                scope="dec_pe")

            ## Dropout
            dec = tf.layers.dropout(dec,
                                    rate=hp.dropout_rate,
                                    training=tf.convert_to_tensor(self.is_training))

            ## Blocks
            for i in range(hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ## Multihead Attention ( self-attention)
                    dec = multihead_attention(queries=dec,
                                              keys=dec,
                                              num_units=hp.hidden_units,
                                              num_heads=hp.num_heads,
                                              dropout_rate=hp.dropout_rate,
                                              is_training=self.is_training,
                                              causality=True,
                                              scope="self_attention")

                    ## Multihead Attention ( vanilla attention)
                    dec = multihead_attention(queries=dec,
                                              keys=enc,
                                              num_units=hp.hidden_units,
                                              num_heads=hp.num_heads,
                                              dropout_rate=hp.dropout_rate,
                                              is_training=self.is_training,
                                              causality=False,
                                              scope="vanilla_attention")

                    ## Feed Forward
                    dec = feedforward(dec, num_units=[4 * hp.hidden_units, hp.hidden_units])

        self.logits = tf.layers.dense(dec, self.target_vocab_size)
        self.preds = tf.to_int32(tf.argmax(self.logits, axis=-1))
        self.istarget = tf.to_float(tf.not_equal(self.y, 0))
        self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y))*self.istarget) / (tf.reduce_sum(self.istarget))
        tf.summary.scalar('acc', self.acc)
        self.params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES
        )
        if self.is_training:
            # Loss
            self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=self.target_vocab_size))
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
            self.mean_loss = tf.reduce_sum(self.loss*self.istarget) / (tf.reduce_sum(self.istarget))
            tf.summary.scalar('mean_loss', self.mean_loss)

            self.global_step = tf.Variable(0, name='global_step')
            # self.gs_op = tf.assign(self.global_step, tf.add(self.global_step, 1))

            self.opt = tf.train.AdamOptimizer(
                learning_rate=hp.learning_rate,
                beta1=0.9,
                beta2=0.98,
                epsilon=1e-8
            ).minimize(self.mean_loss, global_step=self.global_step)
            self.merged = tf.summary.merge_all()


            
        





