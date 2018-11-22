# -*- coding: utf-8 -*-

import logging
from main.utils.vocab import DataVocabs
import os
import pickle
from main.data import Dataset
from config import Config
from main.dl.dlmodel import DLModel
from time import time
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from shutil import copyfile
import torch
import pandas as pd


logger = logging.getLogger("rc")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def prepare(args):
    logger = logging.getLogger("rc")
#    train_test_split(args.all_file, args.train_file, args.test_file, args.train_rate)
    sen_data = Dataset(args)
    sen_data.build_vocab()
    
    with open(args.data_path, 'wb') as fout:
        pickle.dump(sen_data, fout)
    logger.info('Done with preparing!')


def train(args):
    logger = logging.getLogger("rc")
    
    with open(args.data_path, 'rb') as fin:
        dataset = pickle.load(fin)
        
    for tgt_field in args.target_fields:
        model = DLModel(args, dataset.tgt_vocab[tgt_field].size(), dataset.src_vocab.embeddings)
        logger.info('train tgt_field:{}'.format(tgt_field))
        if args.cuda:
            model.cuda()
        best_val_score = 0
        for epoch in range(1, args.epochs + 1):
            model.opts.training = True
            train_batches = dataset.gen_mini_batches('train', args.batch_size, tgt_field)
            for idx, batch in enumerate(train_batches):
                model.update(batch)
                if idx % args.log_per_updates == 0:
                    logger.info('updates[{0:6}] train loss[{1:.5f}]]'.format(
                    model.updates, model.train_loss.avg))
    
            if epoch % args.eval_per_epoch == 0:
                model.opts.training = False
                tgts, preds = [], []
                dev_batches = dataset.gen_mini_batches('dev', args.batch_size, tgt_field,   shuffle=False)
                for batch in dev_batches:
                    pred_pros = model.predict(batch)
                    pred = list(np.argmax(pred_pros, 1))
                    preds.extend(pred)
                    tgts.extend([t['tgt'] for t in batch])
                # print(tgts)
                # print(preds)
                print(classification_report(tgts, preds))
                f1 = f1_score(tgts, preds, average='macro')
                logger.info('f1 :{}'.format(f1))
            
            model_file = os.path.join(args.model_dir,
                                      'tgtfield@{}@epoch@{}@f1@{}.pt'.format(tgt_field,
                                                                             epoch,
                                                                             f1))
            model.save(model_file, epoch)
            if f1 > best_val_score:
                best_val_score = f1
                copyfile(model_file,
                         os.path.join(args.model_dir, tgt_field +'@best_model.pt'))
                logger.info('[new best model saved.]')

def predict(args):
    logger = logging.getLogger("rc")
    df = pd.read_csv(args.test_data_path, encoding='utf-8')
    with open(args.data_path, 'rb') as fin:
        dataset = pickle.load(fin)
        
    for tgt_field in dataset.tgt_info:
        resume_file = tgt_field +'@best_model.pt'
        checkpoint = torch.load(os.path.join(args.model_dir, resume_file))
        state_dict = checkpoint['state_dict']
        model = DLModel(args, 
                        dataset.tgt_vocab[tgt_field].size(), 
                        embedding=dataset.src_vocab.embeddings, 
                        padding_idx=0, 
                        state_dict=state_dict)
        if args.cuda:
            model.cuda()

        model.opts.training = False
        pred_labels = []
        test_batches = dataset.gen_mini_batches('test', args.batch_size, tgt_field, shuffle=False)
        for batch in test_batches:
            pred_pros = model.predict(batch)
            pred = list(np.argmax(pred_pros, 1))
            pred = dataset.tgt_vocab[tgt_field].recover_from_ids(pred)
            pred_labels.extend(pred)
        # print(pred_labels)
        df[tgt_field] = pred_labels
        # print(tgts)
        # print(preds)

    df.to_csv(args.test_data_predict_out_path, encoding='utf-8', index=False)


if __name__ == '__main__':
    pass
#    prepare(Config)
#    train(Config)
#    predict(Config)
