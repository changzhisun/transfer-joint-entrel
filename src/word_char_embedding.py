#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 18/02/07 16:02:57

@author: Changzhi Sun
"""
import sys
sys.path.append("..")

import torch
import torch.nn as nn
from src.char_embedding import CharEmbedding


class WordCharEmbedding(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_size,
                 char_embed_kwargs,
                 dropout=0.5,
                 aux_embedding_size=None,
                 padding_idx=0,
                 concat=False):
        super(WordCharEmbedding, self).__init__()
        self.char_embeddings = CharEmbedding(**char_embed_kwargs)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        if concat and aux_embedding_size is not None:
            ## Only allow aux embedding in concat mode
            self.aux_word_embeddings = nn.Embedding(vocab_size, aux_embedding_size)
        self.concat = concat

    def forward(self, X, X_char=None):
        # Ref: https://github.com/Shawn1993/cnn-text-classification-pytorch/blob/master/model.py
        word_vecs = self.word_embeddings(X)
        if X_char is not None:
            #  char_vecs = torch.cat([
                #  self.char_embeddings(x).unsqueeze(0)
                #  for x in X_char
            #  ], 1)
            batch_size, sent_size, char_size = X_char.size()
            X_char = X_char.view(-1, char_size)
            char_vecs = self.char_embeddings(X_char)
            char_vecs = char_vecs.view(batch_size, sent_size, -1)
            if self.concat:
                embedding_list = [char_vecs, word_vecs]
                if hasattr(self, "aux_word_embeddings"):
                    aux_vecs = self.aux_word_embeddings(X)
                    embedding_list.append(aux_vecs)
                word_vecs = torch.cat(embedding_list, 2)
            else:
                word_vecs = char_vecs + word_vecs
        return self.dropout(word_vecs)

#  char_embed_kwargs = {
        #  "vocab_size" : 10,
        #  "embedding_size": 50,
        #  "out_channels" : 25,
        #  "kernel_sizes" : [2, 3]
    #  }
#  w = WordCharEmbedding(10, 10, char_embed_kwargs, concat=True)
