# -*- coding: utf-8 -*-

class Hyperparams:
    batch_size = 64 # alias = N
    learning_rate = 0.0001 # learning rate. In paper, learning rate is adjusted to the global step.
    source_vocab_file = './vocab.tsv'
    target_vocab_file = './vocab.tsv'
    train_file = './train.txt'
    test_file = './test.txt'
    
    # model
    step_num = 500
    maxlen = 15 # Maximum number of words in a sentence. alias = T.
                # Feel free to increase this if you are ambitious.
    min_cnt = 5 # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512 # alias = C
    num_blocks = 6 # number of encoder/decoder blocks
    num_epochs = 20
    num_heads = 8
    dropout_rate = 0.1
    
    
    
    
