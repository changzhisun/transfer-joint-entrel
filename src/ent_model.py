#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 18/03/03 21:05:12

@author: Changzhi Sun
"""
import torch
import torch.nn as nn
from torch.autograd import Variable

class EntModel(nn.Module):

    def __init__(self,
                 hidden_size,
                 tag_size,
                 use_cuda=False):
        super(EntModel, self).__init__()
        self.use_cuda = use_cuda
        self.hidden_size = hidden_size
        self.tag_size = tag_size

        self.decoder_output = nn.Linear(hidden_size, self.tag_size)
        self.loss_function = nn.CrossEntropyLoss(reduction='sum')
        self.Softmax= nn.Softmax(dim=1)

    def forward(self, rnn_outputs, X_len, Y):
        emissions = self.get_emissions(rnn_outputs)
        loss, pred_tags, probs = self.get_loss_predict_probs(emissions, X_len, Y)
        return loss, pred_tags, probs

    def get_emissions(self, rnn_outputs):
        batch_size = rnn_outputs.size(0)
        emissions = self.decoder_output(rnn_outputs.view(-1, self.hidden_size)) # (batch_size * seq_size) x tag_size
        emissions = emissions.view(batch_size, -1, self.tag_size) # batch_size x seq_size x tag_size
        return emissions

    def get_loss_predict_probs(self, batch_emissions, X_len, Y):
        losses, pred_tags, probs = [], [], []
        for i, emissions in enumerate(batch_emissions):
            emissions = emissions[:X_len[i]]
            y = Variable(Y[i][:X_len[i]])
            loss = self.loss_function(emissions, y)
            _, pred_tag = emissions.max(1)
            if self.use_cuda:
                pred_tag = pred_tag.cpu()
            pred_tag = [e.item() for e in pred_tag]
            pred_tags.append(pred_tag)
            losses.append(loss.unsqueeze(0))
            probs.append(self.Softmax(emissions))
        batch_loss = torch.sum(torch.cat(losses, 0))
        probs = torch.cat(probs, 0)
        return batch_loss, pred_tags, probs
