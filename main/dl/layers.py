# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineLayer(nn.Module):
    def __init__(self):
        super(CosineLayer, self).__init__()
        pass
    
    def forward(self, input_a, input_b):
        input_a_norm = F.normalize(input_a, p=2, dim=1).view(-1, 1, input_a.size(1))
        input_b_norm = F.normalize(input_b, p=2, dim=1).view(-1, 1, input_b.size(1))
        input_b_norm = torch.transpose(input_b_norm, 1, 2)
        x = torch.bmm(input_a_norm, input_b_norm).view(-1, 1)
        return x


class FCLayers(nn.Module):
    def __init__(self, input_size, output_size):
        super(FCLayers, self).__init__()
        self.lin = nn.Linear(input_size, output_size)
        
    def forward(self, batch):
        x = self.lin(batch)
        return x


class TextBiattn(nn.Module):
    def __init__(self, opts, embedding):
        super(TextBiattn, self).__init__()
        self.opts = opts
        
        embedding = torch.Tensor(embedding).float()
        self.embedding = torch.nn.Embedding(embedding.size(0),
                                            embedding.size(1),
                                            padding_idx=0)
        self.embedding.weight.data = embedding
        
        self.encoder = nn.LSTM(input_size=self.opts.embedding_size, 
                              hidden_size=self.opts.hidden_size, 
                              num_layers=self.opts.rnn_layer_num,
                              dropout=self.opts.rnn_dropout_rate,
                              batch_first=True,
                              bidirectional=True)
        
        self.Wc = nn.Linear(2 * self.opts.hidden_size, 1, bias=False)

    def forward(self, batch):
        batch = self.embedding(batch)
        
        h, _ = self.encoder(batch)
        s = self.Wc(h)
        
        ait = F.softmax(s, 1).transpose(2, 1)
        
        qtd = ait.bmm(h)
        
        qtd = qtd.view(-1, 2 * self.opts.hidden_size)
        
        return qtd


class TextCNN(nn.Module):
    def __init__(self, opts, label_num, embedding):
        super(TextCNN, self).__init__()
        self.opts = opts
        self.label_num = label_num
        self.pad_sentence_size = opts.max_sentence_size
        
        embedding = torch.tensor(embedding).float()
        self.embedding = torch.nn.Embedding(embedding.size(0),
                                            embedding.size(1),
                                            padding_idx=0)
        self.embedding.weight.data = embedding
        
        if self.opts.fix_embedding:
            for p in self.embedding.parameters():
                p.requires_grad = False
                
        self.conv_modules = []
        self.build_conv_modules(opts.kernel_sizes)
        # self.dropout = nn.Dropout(p=0.5)
        self.lin = nn.Linear(len(self.conv_modules)*self.opts.out_channels, self.label_num)
        
    def build_conv_modules(self, kernel_sizes):
        for kernel_size in kernel_sizes:
            conv_module = nn.Sequential(
                             nn.Conv2d(in_channels=1,
                                       out_channels=self.opts.out_channels,
                                       kernel_size=(kernel_size, self.opts.embedding_size)),
                             # nn.BatchNorm2d(self.opts.out_channels),
                             nn.ReLU(),
                             nn.MaxPool2d((self.get_feature_map_size(kernel_size, self.pad_sentence_size), 1)),
                             )
            self.conv_modules.append(conv_module)
                
    @staticmethod
    def get_feature_map_size(kernel_size, sentence_size, stride=1, padding=0):
        feature_map_size = (sentence_size + 2*padding - kernel_size) / stride + 1
        return int(feature_map_size)

    def forward(self, batch):
        
        batch = self.embedding(batch)
        batch = batch.view(batch.size(0), 1, self.pad_sentence_size, self.opts.embedding_size)
        
        # batch = batch.float()
        # print(batch)
        
        feature_maps = [conv_module(batch).view(-1, self.opts.out_channels) for
                        conv_module in self.conv_modules]

        x = torch.cat(feature_maps, 1)
        
        # x = self.dropout(x)
        x = F.dropout(x,
                       p=self.opts.dropout_rate,
                       training=self.opts.training)
        x = self.lin(x)
        
        x = x.view(-1, self.label_num)
        
        return x