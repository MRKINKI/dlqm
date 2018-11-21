# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
import torch.nn.functional as F
from .textcnn import TextCNN
from .textbiattn import TextBiattn
from .utils import AverageMeter
import numpy as np


class DLModel:
    def __init__(self, opts, tgt_num, embedding=None, padding_idx=0, state_dict=None):
        

        self.opts = opts
        if opts.model == 'textcnn':
            self.network = TextCNN(opts, tgt_num, embedding)
        elif opts.model == 'textbiattn':
            self.network = TextBiattn(opts, tgt_num, embedding)
            
        if state_dict:
            new_state = set(self.network.state_dict().keys())
            for k in list(state_dict['network'].keys()):
                if k not in new_state:
                    del state_dict['network'][k]
            self.network.load_state_dict(state_dict['network'])
            
#        if state_dict:
#            self.optimizer.load_state_dict(state_dict['optimizer'])
            
        # parameters = [p for p in self.network.parameters() if p.requires_grad]
        # self.optimizer = optim.Adamax(parameters,
        #                        weight_decay=opt['weight_decay'])
        self.optimizer = torch.optim.Adamax(self.network.parameters())
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_loss = AverageMeter()
        self.updates = 0
    
    def update(self, batch):
        self.network.train()
        src = [t['sentence_word_ids'] for t in batch]
        # print([len(t) for t in src])
        tgt = [t['tgt'] for t in batch]
        
        src = torch.LongTensor(src)
        tgt = torch.LongTensor(tgt)
        
        if self.opts.cuda:
            src = src.cuda()
            tgt = tgt.cuda()
        
        self.optimizer.zero_grad()
        output = self.network(src)
        loss = self.criterion(output, tgt)
        
        self.train_loss.update(loss.data[0], src.size(0))
        
        loss.backward()
        self.optimizer.step()
        self.updates += 1
        
    def predict(self, batch):
        self.network.eval()
        src = [t['sentence_word_ids'] for t in batch]
        src = torch.LongTensor(src)
        if self.opts.cuda:
            src = src.cuda()
        output = self.network(src)
        output = output.data.cpu()
        pros = F.softmax(output, dim=1).numpy()
        return pros
    
    def save(self, filename, epoch):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'updates': self.updates
            },
            'epoch': epoch
        }
        torch.save(params, filename)
        
    def cuda(self):
        self.network.cuda()
