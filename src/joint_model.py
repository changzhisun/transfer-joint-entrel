#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 18/02/07 16:33:30

@author: Changzhi Sun
"""
import sys
sys.path.append("..")

import torch
import torch.nn as nn
import numpy as np

from src.ent_model import EntModel
from src.rel_model import RelModel

class JointModel(nn.Module):

    def __init__(self,
                 word_char_embedding,
                 word_char_emb_size,
                 hidden_size,
                 ent_kwargs_list,
                 rel_kwargs_list,
                 ent_span_kwargs,
                 rel_bin_kwargs,
                 num_layers=1,
                 use_cuda=False,
                 bidirectional=True,
                 add_share_loss=True,
                 add_trans_loss=True,
                 sch_k=0.5,
                 dropout=0.5):
        super(JointModel, self).__init__()
        self.word_char_embedding = word_char_embedding
        self.word_char_emb_size = word_char_emb_size
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda
        self.sch_k = sch_k
        self.add_share_loss = add_share_loss
        self.add_trans_loss = add_trans_loss

        self.ent_span_model = EntModel(**ent_span_kwargs)
        self.rel_bin_model = RelModel(**rel_bin_kwargs)

        self.ent_models = nn.ModuleList(
            [EntModel(**kwargs) for kwargs in ent_kwargs_list])
        self.rel_models = nn.ModuleList(
            [RelModel(**kwargs) for kwargs in rel_kwargs_list])

        self.share_rnn = nn.LSTM(word_char_emb_size,
                                self.hidden_size // 2,
                                num_layers=num_layers,
                                bidirectional=bidirectional,
                                batch_first=True,
                                dropout=dropout)
        self.private_rnns = nn.ModuleList(
            [nn.LSTM(word_char_emb_size,
                     self.hidden_size // 2,
                     num_layers=num_layers,
                     bidirectional=bidirectional,
                     batch_first=True,
                     dropout=dropout)
             for _ in ent_kwargs_list])
        self.ent2span = nn.ModuleList(
            [nn.Linear(self.ent_models[i].tag_size,
                       self.ent_span_model.tag_size,
                       bias=False)
             for i in range(len(ent_kwargs_list))])

        self.dropout = nn.Dropout(dropout)

    def forward(self, X, X_char, X_len, Y_span, Y, Y_rel, dom_id, i_epoch=0):
        batch_wc_embs, share_rnn_outputs = self.run_rnn(X, X_char, X_len, self.share_rnn)
        share_rnn_outputs = share_rnn_outputs.contiguous()
        
        # Share Model
        if self.add_share_loss:
            ent_span_loss, pred_ent_span_tags, pred_ent_span_probs = self.ent_span_model(share_rnn_outputs, X_len, Y_span)
            if self.training:
                sch_ent_span_tags = self.schedule_sample(pred_ent_span_tags, Y_span, i_epoch)
            else:
                sch_ent_span_tags = pred_ent_span_tags
            rel_bin_loss, pred_rel_bin_tags, candi_rel_bin_num = self.rel_bin_model(batch_wc_embs,
                                                                                    share_rnn_outputs,
                                                                                    sch_ent_span_tags,
                                                                                    Y_rel,
                                                                                    X_len)


        # Private Model
        _, private_rnn_outputs = self.run_rnn(X, X_char, X_len, self.private_rnns[dom_id])
        batch_rnn_outputs = torch.cat([share_rnn_outputs, private_rnn_outputs], 2)

        ent_loss, pred_ent_tags, pred_ent_probs = self.ent_models[dom_id](batch_rnn_outputs, X_len, Y)
        if self.training:
            sch_ent_tags = self.schedule_sample(pred_ent_tags, Y, i_epoch)
        else:
            sch_ent_tags = pred_ent_tags
        rel_loss, pred_rel_tags, candi_rel_num = self.rel_models[dom_id](
            batch_wc_embs, batch_rnn_outputs, sch_ent_tags, Y_rel, X_len)
        if self.add_share_loss:
            if not self.add_trans_loss:
                return (ent_loss, ent_span_loss, rel_loss, rel_bin_loss,
                        pred_ent_tags, pred_ent_span_tags, pred_rel_tags, pred_rel_bin_tags,
                        candi_rel_num, candi_rel_bin_num)

            trans_loss = self.get_trans_loss(self.ent2span[dom_id](pred_ent_probs).view(-1),
                                             pred_ent_span_probs.view(-1))
            return (ent_loss, ent_span_loss, rel_loss, rel_bin_loss,
                    pred_ent_tags, pred_ent_span_tags, pred_rel_tags, pred_rel_bin_tags,
                    candi_rel_num, candi_rel_bin_num, trans_loss)

        return (ent_loss, rel_loss, 
                pred_ent_tags, pred_rel_tags, candi_rel_num)

    def get_trans_loss(self, pred0, pred1):
        return torch.sum((pred0 - pred1) ** 2) / len(pred0)

    def run_rnn(self, X, X_char, X_len, rnn_model):
        batch_size = X.size(0)
        seq_size = X.size(1)
        embed = self.word_char_embedding(X, X_char) # batch_size x seq_size x embed_size
        embed_pack = nn.utils.rnn.pack_padded_sequence(embed, X_len, batch_first=True)
        encoder_outputs, _ = rnn_model(embed_pack) # batch_size x seq_size x hidden_size
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(encoder_outputs, batch_first=True) # batch_size x seq_size x hidden_size
        encoder_outputs = self.dropout(encoder_outputs)
        return embed, encoder_outputs

    def schedule_sample(self, pred_tags, Y, i_epoch):
        sch_p = self.sch_k / (self.sch_k + np.exp(i_epoch / self.sch_k))
        sch_tags = []
        for i, tags in enumerate(pred_tags):
            each_tags = []
            for j, tag in enumerate(tags):
                rd = np.random.random()
                if rd <= sch_p:
                    each_tags.append(Y[i][j])
                else:
                    each_tags.append(tag)
            sch_tags.append(each_tags)
        return sch_tags
