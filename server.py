# -*- coding: utf-8 -*-

from main.dl.dlmodel import DLModel
import torch
from config import Config as args
from main.data import Dataset
import pickle

with open(args.data_path, 'rb') as fin:
    dataset = pickle.load(fin)

resume_file = './data/model/match@best_model.pt'
checkpoint = torch.load(resume_file)
tgt_field = 'match'


state_dict = checkpoint['state_dict']
model = DLModel(args, 
                dataset.tgt_vocab_dict[tgt_field].size(), 
                embedding=dataset.src_vocab.embeddings, 
                padding_idx=0, 
                state_dict=state_dict)

if args.cuda:
    model.cuda()

    
model.opts.training = False

sample = {}

sample['q1_tokens'] = list('牛肉有什么禁忌')

sample['q2_tokens'] = list('什么人不能吃牛肉')

sample['q1_token_ids'] = dataset.src_vocab.convert_to_ids(sample['q1_tokens'])

sample['q2_token_ids'] = dataset.src_vocab.convert_to_ids(sample['q2_tokens'])

batch = [sample]

pred_pros = model.predict(batch)

print(pred_pros)
