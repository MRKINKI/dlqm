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
    model_dir = './data/model'
    
    epochs = 20
    batch_size = 32
    log_per_updates = 10
    eval_per_epoch = 1
    fc_layer_dropout_rate = 0.1
    
    # text_biattn
    cuda = False
    hidden_size = 128
    rnn_layer_num = 1
    rnn_dropout_rate = 0.3
    
    tfidf_threshold = 0.5
    
    match_module_type = 'cos'