import torch
import torch.nn as nn
from .layers import TextBiattn, FCLayers, CosineLayer
import torch.nn.functional as F


class TextMatcher(nn.Module):
    def __init__(self, opts, embedding):
        super(TextMatcher, self).__init__()
        self.opts = opts
        self.vectorizer = TextBiattn(opts, embedding)
        self.fc = FCLayers(4 * self.opts.hidden_size, 2)
        self.cos = CosineLayer()

    def forward(self, q1_batch, q2_batch):
        q1_vector_batch = self.vectorizer(q1_batch)
        q2_vector_batch = self.vectorizer(q2_batch)
        
        if self.opts.match_module_type == 'mlp':
            concat_batch = torch.cat([q1_vector_batch, q2_vector_batch], 1)
            
            # concat_batch = F.dropout(concat_batch,
            #                          p=self.opts.fc_layer_dropout_rate,
            #                          training=self.opts.training)
            
            x = self.fc(concat_batch)
            x = x.view(-1, 2)
        elif self.opts.match_module_type == 'cos':
            x = self.cos(q1_vector_batch, q2_vector_batch)
            # print(x)
            tmp = torch.ones(x.size(0), 1)
            tmp = tmp - x
            x = torch.cat([tmp, x], 1)
        
        return x
        