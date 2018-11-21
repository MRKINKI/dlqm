# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


class TextBiattn(nn.Module):
    def __init__(self, opts, label_num, embedding):
        super(TextBiattn, self).__init__()
        self.opts = opts
        self.label_num = label_num
        self.pad_sentence_size = opts.max_sentence_size
        
        embedding = torch.tensor(embedding).float()
        self.embedding = torch.nn.Embedding(embedding.size(0),
                                            embedding.size(1),
                                            padding_idx=0)
        self.embedding.weight.data = embedding
        
        self.encoder = nn.GRU(input_size=self.opts.embedding_size, 
                              hidden_size=self.opts.hidden_size, 
                              num_layers=self.opts.rnn_layer_num,
                              dropout=self.opts.rnn_dropout_rate,
                              batch_first=True,
                              bidirectional=True)
        
        self.Wc = nn.Linear(2 * self.opts.hidden_size, 1, bias=False)
        
        self.lin = nn.Linear(2 * self.opts.hidden_size, self.label_num)
        

    def forward(self, batch):
        batch = self.embedding(batch)
        
        h, _ = self.encoder(batch)
        s = self.Wc(h)
        
        ait = F.softmax(s, 1).transpose(2, 1)
        
        qtd = ait.bmm(h)
        
        qtd = qtd.view(-1, 2 * self.opts.hidden_size)
        # x = self.dropout(x)

        x = self.lin(qtd)
        
        x = x.view(-1, self.label_num)
        
        return x