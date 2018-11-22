# -*- coding: utf-8 -*-

class Config:
    all_train_data_path = './data/mqa.csv'
    embedding_size = 100
    min_count = 2
    target_fields = ['match']
    fix_sentence_size = False
    max_sentence_size = 30
    create_dev = True
    dev_rate = 0.1
    train_data_path = './data/train.csv'
    dev_data_path = './data/test.csv'
    embedding_path = './data/emb.txt'
    data_path = './data/sen.dat'