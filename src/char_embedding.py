#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 18/02/07 16:01:54

@author: Changzhi Sun
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CharEmbedding(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_size,
                 out_channels,
                 kernel_sizes,
                 padding_idx=0,
                 dropout=0.5):
        super(CharEmbedding, self).__init__()
        self.char_embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        # Usage of nn.ModuleList is important
        ## See: https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/6
        self.convs1 = nn.ModuleList([nn.Conv2d(1, out_channels, (K, embedding_size), padding=(K-1, 0))
                                     for K in kernel_sizes])
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        x = self.char_embeddings(X)
        x = self.dropout(x)
        # Ref: https://github.com/Shawn1993/cnn-text-classification-pytorch/blob/master/model.py
        x = x.unsqueeze(1) # (N,Ci,W,D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] #[(N,Co,W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] #[(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        return self.dropout(x)

