#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 18/05/08 16:22:37

@author: Changzhi Sun
"""

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

from lib.utils import parse_tag
from lib.utils import start_of_chunk
from lib.utils import end_of_chunk
from lib.utils import convert_float_variable
from lib.utils import convert_long_variable
from lib.utils import assign_embeddings


class RelFeatExtractor(nn.Module):

    def __init__(self,
                 hidden_size,
                 out_channels,
                 kernel_sizes,
                 max_sent_len,
                 use_cuda,
                 dropout=0.5):
        super(RelFeatExtractor, self).__init__()
        self.hidden_size = hidden_size
        self.conv_input_size = hidden_size
        self.max_sent_len = max_sent_len
        self.dropout = nn.Dropout(dropout)
        self.use_cuda = use_cuda
        self.b_convs = nn.ModuleList([nn.Conv2d(1,
                                                out_channels,
                                                (K, self.conv_input_size),
                                                padding=(K-1, 0))
                                      for K in kernel_sizes])
        self.mid_convs = nn.ModuleList([nn.Conv2d(1,
                                                  out_channels,
                                                  (K, self.conv_input_size),
                                                  padding=(K-1, 0))
                                        for K in kernel_sizes])
        self.e_convs = nn.ModuleList([nn.Conv2d(1,
                                                out_channels,
                                                (K, self.conv_input_size),
                                                padding=(K-1, 0))
                                      for K in kernel_sizes])
        self.rel_input_size = len(kernel_sizes) * out_channels * 3 + max_sent_len + 2 * self.hidden_size
        self.norm = nn.LayerNorm(self.hidden_size)
        self.output_seq = nn.Sequential(nn.Linear(self.rel_input_size,
                                                  self.hidden_size),
                                        self.norm,
                                        nn.ReLU(),
                                        self.dropout)

    def forward(self,
                Z,
                candi_ent_idxs,
                idx2batch,
                batch_rnn_outputs,
                X_len):

        final_vecs = self.get_final_vecs(candi_ent_idxs,
                                         idx2batch,
                                         Z,
                                         batch_rnn_outputs,
                                         X_len)
        output = self.output_seq(final_vecs)
        return output

    def get_final_vecs(self,
                       candi_ent_idxs,
                       idx2batch,
                       Z,
                       batch_rnn_outputs,
                       X_len):
        b_vecs, mid_vecs, e_vecs = self.get_conv_feature(candi_ent_idxs,
                                                         idx2batch,
                                                         Z)
        dist_vecs = self.get_distance_between_entity(candi_ent_idxs)

        left_vecs, right_vecs = self.get_segment_feature(candi_ent_idxs,
                                                         Z,
                                                         X_len,
                                                         idx2batch)
        final_vecs = [mid_vecs, b_vecs, e_vecs, dist_vecs, left_vecs, right_vecs]
        final_vecs = torch.cat(final_vecs, 1)
        return final_vecs

    def get_conv_feature(self, candi_ent_idxs, idx2batch, Z):
        b_vecs = []
        mid_vecs = []
        e_vecs = []
        for i in range(len(candi_ent_idxs)):
            batch_idx = idx2batch[i]
            z = Z[batch_idx]
            b, e = candi_ent_idxs[i]
            assert b[0] < e[0]
            if b[-1] + 1 == e[0]:
                mid_vecs.append([])
            else:
                mid_vecs.append(list(z[b[-1]+1: e[0]].split(1)))
            b_vecs.append(list(z[b[0]: b[-1]+1].split(1)))
            e_vecs.append(list(z[e[0]: e[-1]+1].split(1)))

        mid_vecs = self.pad_feature(mid_vecs)
        b_vecs = self.pad_feature(b_vecs)
        e_vecs = self.pad_feature(e_vecs)

        mid_vecs = self.get_conv(mid_vecs, self.mid_convs)
        b_vecs = self.get_conv(b_vecs, self.b_convs)
        e_vecs = self.get_conv(e_vecs, self.e_convs)
        return b_vecs, mid_vecs, e_vecs

    def get_conv(self, h, convs):
        h = self.dropout(h)
        h = h.unsqueeze(1) # batch_size x 1 x seq_size x conv_input_size
        h = [F.relu(conv(h)).squeeze(3) for conv in convs] #[(N,Co,W), ...]*len(Ks)
        h = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in h] #[(N,Co), ...]*len(Ks)
        h = torch.cat(h, 1)
        return h

    def pad_feature(self, features):
        max_len = max([len(e) for e in features])

        def init_pad_h():
            pad_h = Variable(torch.zeros(1, self.conv_input_size))
            if self.use_cuda:
                pad_h = pad_h.cuda(async=True)
            return pad_h

        if max_len == 0:
            return torch.cat([init_pad_h() for _ in features], 0).unsqueeze(1)
        f = []
        for feature in features:
            feature = feature + [init_pad_h() for e in range(max_len - len(feature))]
            feature = torch.cat(feature, 0) # seq_size x conv_input_size
            f.append(feature.unsqueeze(0)) # 1 x seq_size x conv_input_size
        return torch.cat(f, 0) # batch_size x seq_size x conv_input_size

    def get_distance_between_entity(self, candi_ent_idxs):
        dist_vecs = []
        for i in range(len(candi_ent_idxs)):
            b, e = candi_ent_idxs[i]
            assert b[0] < e[0]
            distance = np.eye(self.max_sent_len)[e[0] - b[-1]]
            distance = convert_float_variable(distance, self.use_cuda).unsqueeze(0)
            dist_vecs.append(distance)
        dist_vecs = torch.cat(dist_vecs, 0)
        return dist_vecs

    def get_forward_segment(self, fward_rnn_output, b, e):
        max_len, h_size = fward_rnn_output.size()
        if b > e:
            zero_vec = convert_float_variable(torch.zeros(h_size), self.use_cuda)
            return zero_vec
        if b == 0:
            return fward_rnn_output[e]
        return fward_rnn_output[e] - fward_rnn_output[b - 1]

    def get_backward_segment(self, bward_rnn_output, b, e):
        max_len, h_size = bward_rnn_output.size()
        if b > e:
            zero_vec = convert_float_variable(torch.zeros(h_size), self.use_cuda)
            return zero_vec
        if e == max_len - 1:
            return bward_rnn_output[b]
        return bward_rnn_output[b] - bward_rnn_output[e + 1]

    def get_segment_feature(self,
                            candi_ent_idxs,
                            batch_rnn_outputs,
                            X_len,
                            idx2batch):
        left_vecs = []
        right_vecs = []
        hidden_size = self.hidden_size
        for i in range(len(candi_ent_idxs)):
            b, e = candi_ent_idxs[i]
            batch_idx = idx2batch[i]
            cur_len = X_len[batch_idx]
            rnn_outputs = batch_rnn_outputs[batch_idx]

            fward_rnn_output, bward_rnn_output = rnn_outputs.split(hidden_size // 2, 1)

            fward_left_vec = self.get_forward_segment(fward_rnn_output, 0, b[0] - 1)
            bward_left_vec = self.get_backward_segment(bward_rnn_output, 0, b[0] - 1)
            left_vec = torch.cat([fward_left_vec, bward_left_vec], 0).unsqueeze(0)
            left_vecs.append(left_vec)

            fward_right_vec = self.get_forward_segment(fward_rnn_output, e[-1] + 1, cur_len - 1)
            bward_right_vec = self.get_forward_segment(bward_rnn_output, e[-1] + 1, cur_len - 1)
            right_vec = torch.cat([fward_right_vec, bward_right_vec], 0).unsqueeze(0)
            right_vecs.append(left_vec)
        left_vecs = torch.cat(left_vecs, 0)
        right_vecs = torch.cat(right_vecs, 0)
        return left_vecs, right_vecs
