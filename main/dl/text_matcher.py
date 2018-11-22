import torch
import torch.nn as nn
from .layers import TextBiattn, FCLayers
import torch.nn.functional as F


class TextMatcher(nn.Module):
    def __init__(self, opts, embedding):
        super(TextMatcher, self).__init__()
        self.opts = opts
        self.vectorizer = TextBiattn(opts, embedding)
        self.fc = FCLayers(4 * self.opts.hidden_size, 2)
        

    def forward(self, q1_batch, q2_batch):
        q1_vector_batch = self.vectorizer(q1_batch)
        q2_vector_batch = self.vectorizer(q2_batch)

        concat_batch = torch.cat([q1_vector_batch, q2_vector_batch], 1)
        # q1_vertor_batch.mm(q2_vector_batch)
        
        # concat_batch = F.dropout(concat_batch,
        #                          p=self.opts.fc_layer_dropout_rate,
        #                          training=self.opts.training)
        
        x = self.fc(concat_batch)
        x = x.view(-1, 2)
        
        return x
        