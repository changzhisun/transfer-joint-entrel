#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 18/03/03 21:06:14

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

class RelModel(nn.Module):

    def __init__(self,
                 hidden_size,
                 chunk_vocab,
                 N_ID,
                 out_channels,
                 kernel_sizes,
                 rel_size,
                 max_sent_len,
                 use_cuda=False,
                 dropout=0.5,
                 is_bin_rel=False):
        super(RelModel, self).__init__()
        self.N_ID = N_ID
        self.max_sent_len = max_sent_len
        self.rel_size = rel_size
        self.use_cuda = use_cuda
        self.chunk_vocab = chunk_vocab
        self.ent_tag_size = chunk_vocab.size
        self.is_bin_rel = is_bin_rel

        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.conv_input_size = self.hidden_size + self.ent_tag_size

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
        self.Tanh = nn.Tanh()
        self.Softmax= nn.Softmax()

        self.output_seq = nn.Sequential(nn.Linear(self.rel_input_size,
                                                  self.hidden_size),
                                        nn.ReLU(),
                                        self.dropout,
                                        nn.Linear(self.hidden_size,
                                                  rel_size))
        self.loss_function = nn.CrossEntropyLoss(reduction='sum')
        #  self.loss_function = nn.MultiMarginLoss()
        self.ent_emb_oh = nn.Embedding(self.ent_tag_size, self.ent_tag_size)
        assign_embeddings(self.ent_emb_oh, np.eye(self.ent_tag_size), fix_embedding=True)

        self.chunk_num, self.idx2chunk = self.parse_chunk_vocab(chunk_vocab)
        self.chunk2idx = {v : k for k, v in enumerate(self.idx2chunk)}

    def parse_chunk_vocab(self, chunk_vocab):
        entity_set = set()
        for tag in chunk_vocab.idx2item:
            if tag == 'O':
                continue
            _, tag_type = parse_tag(tag)
            entity_set.add(tag_type)
        entity_id2item = list(entity_set)
        return len(entity_id2item), entity_id2item

    def forward(self,
                batch_wc_embs,
                batch_rnn_outputs,
                batch_pred_ent_tags,
                Y_rel,
                X_len):
        batch_size = len(X_len)
        candi_ent_idxs, labels, idx2batch = self.generate_candidate_entity_pair(batch_pred_ent_tags, Y_rel, X_len)
        if len(candi_ent_idxs) == 0:
            loss = convert_float_variable([0], self.use_cuda)
            loss.requires_grad = True
            return loss, [[] for _ in range(batch_size)], 0

        scores = self.get_score(candi_ent_idxs,
                                idx2batch,
                                batch_pred_ent_tags,
                                batch_wc_embs,
                                batch_rnn_outputs,
                                X_len)
        batch_rel_loss, batch_pred_rel_tags = self.get_loss_and_predict(scores,
                                                                        labels,
                                                                        batch_size,
                                                                        idx2batch,
                                                                        candi_ent_idxs)
        return batch_rel_loss, batch_pred_rel_tags, len(candi_ent_idxs)

    def generate_relation_dict(self, y_rel):
        rel_dict = {}
        for b, e, t in y_rel:
            b = tuple(range(b[0], b[-1]))
            e = tuple(range(e[0], e[-1]))
            rel_dict[(b, e)] = t
        return rel_dict

    def get_entity_idx2chunk_type(self, t_entity):
        entity_idx2chunk_type = {}
        for k, v in t_entity.items():
            for e in v:
                entity_idx2chunk_type[e] = self.chunk2idx[k]
        return entity_idx2chunk_type

    def generate_candidate_entity_pair_with_win(self, entity_idx2chunk_type):
        instance_candidate_set = set()
        for ent1_idx in entity_idx2chunk_type.keys():
            for ent2_idx in entity_idx2chunk_type.keys():
                if ent1_idx[0] >= ent2_idx[0]:
                    continue
                instance_candidate_set.add((ent1_idx, ent2_idx))
        return instance_candidate_set

    def add_gold_candidate(self, instance_candidate_set, y_rel):
        for b, e, t in y_rel:
            b = tuple(range(b[0], b[-1]))
            e = tuple(range(e[0], e[-1]))
            if set(b) & set(e) == set():
                instance_candidate_set.add((b, e))

    def adjust_negative_ratio(self, instance_candidate_set, rel_dict, r=1.0):
        position_num = len(rel_dict)
        negative_num = len(instance_candidate_set) - position_num

        if negative_num <= r * position_num:
            return instance_candidate_set

        negative_num = int(r * position_num)
        negative_instance_list = []
        positive_instance_list = []
        for b, e in instance_candidate_set:
            if (b, e) in rel_dict:
                positive_instance_list.append((b, e))
            else:
                negative_instance_list.append((b, e))
        np.random.shuffle(negative_instance_list)
        negative_instance_list = negative_instance_list[:negative_num]
        return set(positive_instance_list) | set(negative_instance_list)

    def generate_candidate_entity_pair(self, Y, Y_rel, X_len):
        labels = []
        idx2batch = {}
        entity_pair_idxs = []
        for batch_idx in range(len(Y)):
            cur_len = X_len[batch_idx]

            rel_dict = self.generate_relation_dict(Y_rel[batch_idx])
            y = Y[batch_idx][:cur_len]

            y = [self.chunk_vocab.idx2item[t] for t in y]
            t_entity = self.get_entity(y)

            entity_idx2chunk_type = self.get_entity_idx2chunk_type(t_entity)

            instance_candidate_set = self.generate_candidate_entity_pair_with_win(entity_idx2chunk_type)

            if self.training:
                self.add_gold_candidate(instance_candidate_set, Y_rel[batch_idx])
                #  instance_candidate_set = self.adjust_negative_ratio(instance_candidate_set, rel_dict)

            for b, e in instance_candidate_set:
                if (b, e) in rel_dict:
                    if self.is_bin_rel:
                        t = 1
                    else:
                        t = rel_dict[(b, e)]
                else:
                    t = self.N_ID
                idx2batch[len(entity_pair_idxs)] = batch_idx
                entity_pair_idxs.append((b, e))
                labels.append(t)
        return entity_pair_idxs, labels, idx2batch

    def generate_word_representation(self, Y, X_len, batch_wc_embs, batch_rnn_outputs):
        Z = []
        for batch_idx in range(len(Y)):
            x_len = X_len[batch_idx]
            y = Y[batch_idx][: x_len] # seq_size x 1

            y = convert_long_variable(y, self.use_cuda)
            y_embs = self.ent_emb_oh(y)

            rnn_outputs = batch_rnn_outputs[batch_idx][: x_len]
            wc_embs = batch_wc_embs[batch_idx][: x_len]

            #  Z.append(torch.cat([wc_embs, rnn_outputs, y_embs], 1))
            Z.append(torch.cat([rnn_outputs, y_embs], 1))
        return Z

    def get_score(self,
                  candi_ent_idxs,
                  idx2batch,
                  Y,
                  batch_wc_embs,
                  batch_rnn_outputs,
                  X_len):
        Z = self.generate_word_representation(Y,
                                              X_len,
                                              batch_wc_embs,
                                              batch_rnn_outputs)
        final_vecs = self.get_final_vecs(candi_ent_idxs,
                                         idx2batch,
                                         Z,
                                         batch_rnn_outputs,
                                         X_len)
        scores = self.output_seq(final_vecs)
        return scores

    def get_conv_feature(self, candi_ent_idxs, idx2batch, Z, batch_rnn_outputs):
        b_vecs = []
        mid_vecs = []
        e_vecs = []
        for i in range(len(candi_ent_idxs)):
            batch_idx = idx2batch[i]
            z = Z[batch_idx]
            rnn_outputs = batch_rnn_outputs[batch_idx]
            b, e = candi_ent_idxs[i]

            assert b[0] < e[0]

            if b[-1] + 1 == e[0]:
                mid_vecs.append([])
            else:
                #  mid_vecs.append(list(rnn_outputs[b[-1]+1: e[0]].split(1)))
                mid_vecs.append(list(z[b[-1]+1: e[0]].split(1)))

            b_vecs.append(list(z[b[0]: b[-1]+1].split(1)))
            e_vecs.append(list(z[e[0]: e[-1]+1].split(1)))

        #  mid_vecs = self.pad_feature_with_tag(mid_vecs)
        mid_vecs = self.pad_feature(mid_vecs)
        b_vecs = self.pad_feature(b_vecs)
        e_vecs = self.pad_feature(e_vecs)

        mid_vecs = self.get_conv(mid_vecs, self.mid_convs)
        b_vecs = self.get_conv(b_vecs, self.b_convs)
        e_vecs = self.get_conv(e_vecs, self.e_convs)
        return b_vecs, mid_vecs, e_vecs

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

    def get_final_vecs(self,
                       candi_ent_idxs,
                       idx2batch,
                       Z,
                       batch_rnn_outputs,
                       X_len):
        b_vecs, mid_vecs, e_vecs = self.get_conv_feature(candi_ent_idxs,
                                                         idx2batch,
                                                         Z,
                                                         batch_rnn_outputs)
        dist_vecs = self.get_distance_between_entity(candi_ent_idxs)

        left_vecs, right_vecs = self.get_segment_feature(candi_ent_idxs,
                                                         batch_rnn_outputs,
                                                         X_len,
                                                         idx2batch)
        #  final_vecs = [relative_pos_vecs, mid_vecs, b_vecs, e_vecs, dist_vecs]
        final_vecs = [mid_vecs, b_vecs, e_vecs, dist_vecs, left_vecs, right_vecs]
        #  final_vecs = [mid_vecs, b_vecs, e_vecs, dist_vecs]
        final_vecs = torch.cat(final_vecs, 1)
        return final_vecs

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

    def pad_feature_with_tag(self, features):
        max_len = max([len(e) for e in features])

        def init_pad_h():
            pad_h = Variable(torch.zeros(1, self.conv_input_size - self.ent_tag_size))
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

    def get_loss_and_predict(self,
                             scores,
                             labels,
                             batch_size,
                             idx2batch,
                             candi_ent_idxs):
        labels = convert_long_variable(labels, self.use_cuda)
        _, max_i = scores.max(1)
        if self.use_cuda:
            max_i = max_i.cpu()
        batch_rel_loss = self.loss_function(scores, labels)
        pred_rel_tags = [[] for _ in range(batch_size)]
        for i, (b, e) in enumerate(candi_ent_idxs):
            cur_i = max_i[i].data.numpy()
            if cur_i != self.N_ID:
                pred_rel_tags[idx2batch[i]].append((b, e, cur_i))
        return batch_rel_loss, pred_rel_tags

    def get_entity(self, y):
        last_guessed = 'O'        # previously identified chunk tag
        last_guessed_type = ''    # type of previous chunk tag in corpus
        guessed_idx = []
        t_guessed_entity2idx = defaultdict(list)
        for i, tag in enumerate(y):
            guessed, guessed_type = parse_tag(tag)
            start_guessed = start_of_chunk(last_guessed, guessed,
                                                last_guessed_type, guessed_type)
            end_guessed = end_of_chunk(last_guessed, guessed,
                                            last_guessed_type, guessed_type)
            if start_guessed:
                if guessed_idx:
                    t_guessed_entity2idx[guessed_idx[0]].append(tuple(guessed_idx[1:]))
                guessed_idx = [guessed_type, i]
            elif guessed_idx and not start_guessed and guessed_type == guessed_idx[0]:
                guessed_idx.append(i)

            last_guessed = guessed
            last_guessed_type = guessed_type
        if guessed_idx:
            t_guessed_entity2idx[guessed_idx[0]].append(tuple(guessed_idx[1:]))
        return t_guessed_entity2idx

