# -*- coding: utf-8 -*-
import pandas as pd
# import logging
from .utils.tokenize import Tokenizer
from .utils.vocab import Vocab
import numpy as np
from sklearn.model_selection import train_test_split
from multiprocessing.pool import ThreadPool
from multiprocessing.dummy import Pool
from time import time


class Dataset(object):

    def __init__(self, opts):
        
        self.opts = opts
        self.max_sentence_size = opts.max_sentence_size
        self.min_count = opts.min_count
        self.embedding_size = opts.embedding_size
        
        self.tokenizer = Tokenizer()
        
        self.train_set, self.dev_set, self.test_set = [], [], []
        
        self.target_fields = opts.target_fields
        
        if opts.create_dev:
            self.train_dev_split()

        self.train_set = self._load_dataset(opts.train_data_path, train=True)

        self.dev_set = self._load_dataset(opts.dev_data_path)

        # self.test_set = self._load_dataset(opts.test_data_path)

        self.src_vocab = None
        self.tgt_vocab_dict = None
        
    def train_dev_split(self):
        df = pd.read_csv(self.opts.all_train_data_path, encoding='utf-8')
        train, dev = train_test_split(df, test_size=self.opts.dev_rate)
        train.to_csv(self.opts.train_data_path)
        dev.to_csv(self.opts.dev_data_path)
        
    def get_tokens(self, sample):
        sample['q1_tokens'] = self.tokenizer.tokenize(sample['q1'])
        sample['q2_tokens'] = self.tokenizer.tokenize(sample['q2'])
        
    def _load_dataset(self, data_path, header=0, train=False):
        samples = pd.read_csv(data_path, header=header, encoding='utf-8').to_dict('records')

        for sample in samples:
            self.get_tokens(sample)
        return samples

    def build_vocab(self):
        src_vocab = Vocab()
        for word in self.word_iter('train'):
            src_vocab.add(word)

        src_vocab.filter_tokens_by_cnt(min_cnt=self.min_count)

        # src_vocab.randomly_init_embeddings(self.embedding_size)
        src_vocab.load_pretrained_embeddings(self.opts.embedding_path, self.opts.embedding_size)
        
        tgt_vocab_dict = {}
        for tgt_field in self.target_fields:
            tgt_vocab = Vocab(initial_tokens=False, lower=False)
            for sample in self.train_set:
                tgt = sample[tgt_field]
                tgt_vocab.add(tgt)
            tgt_vocab_dict[tgt_field] = tgt_vocab
        self.src_vocab = src_vocab
        self.tgt_vocab_dict = tgt_vocab_dict
        self.convert_to_ids()
    
    def _one_mini_batch(self, data, indices, tgt_field, set_name):

        raw_data = [data[i] for i in indices]
        batch = []
        for sidx, sample in enumerate(raw_data):
            batch_data = {}
            batch_data['q1_token_ids'] = sample['q1_token_ids']
            batch_data['q2_token_ids'] = sample['q2_token_ids']
            if set_name in ['train', 'dev']:
                batch_data['tgt'] = sample[tgt_field + '_id']

            batch.append(batch_data)
        self._dynamic_padding(batch, 0, 'q1_token_ids')
        self._dynamic_padding(batch, 0, 'q2_token_ids')
        return batch

    def _dynamic_padding(self, batch_data, pad_id, field):
        
        if self.opts.fix_sentence_size:
            pad_sentence_size = self.max_sentence_size
        else:
            pad_sentence_size = min(self.max_sentence_size, 
                                max([len(t[field]) for t in batch_data]))
            
        
        for sub_batch_data in batch_data:
            ids = sub_batch_data[field]
            # print(len(ids), pad_sentence_size)
            sub_batch_data[field] = ids + [pad_id] * (pad_sentence_size - len(ids))
            sub_batch_data[field] = sub_batch_data[field][:pad_sentence_size]
            # print(len(sub_batch_data['sentence_word_ids'] ))
        return batch_data

    def word_iter(self, set_name=None):

        if set_name is None:
            data_set = self.train_set + self.dev_set + self.test_set
        elif set_name == 'train':
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        if data_set is not None:
            for sample in data_set:
                for token in sample['q1_tokens']:
                    yield token
                for token in sample['q2_tokens']:
                    yield token
                    

    def convert_to_ids(self):

        for idx, data_set in enumerate([self.train_set, self.dev_set, self.test_set]):
            if not len(data_set):
                continue
            for sample in data_set:
                sample['q1_token_ids'] = self.src_vocab.convert_to_ids(sample['q1_tokens'])
                sample['q2_token_ids'] = self.src_vocab.convert_to_ids(sample['q2_tokens'])
                if idx <= 1:
                    for tgt_field in self.target_fields:
                        if tgt_field in sample:
                            sample[tgt_field + '_id'] = self.tgt_vocab_dict[tgt_field].get_id(sample[tgt_field])

    def gen_mini_batches(self, set_name, batch_size, tgt_field, shuffle=True):

        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
        elif set_name == 'test':
            data = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        data_size = len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(data, batch_indices, tgt_field, set_name)