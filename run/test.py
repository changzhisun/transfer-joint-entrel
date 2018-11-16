#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 18/03/14 16:46:36

@author: Changzhi Sun
"""
import os
import sys
sys.path.append('..')
import argparse
import json

import torch
import torch.optim as optim
import numpy as np

from lib import vocab, utils
from src.word_char_embedding import WordCharEmbedding
from src.joint_model import JointModel
from entrel_eval import eval_file
from config import Configurable

torch.manual_seed(1) # CPU random seed
np.random.seed(1)

argparser = argparse.ArgumentParser()
argparser.add_argument('--config_file', default='../configs/default.cfg')
#  argparser.add_argument('--model', default='BaseParser')
args, extra_args = argparser.parse_known_args()
config = Configurable(args.config_file, extra_args)

use_cuda = config.use_cuda

# GPU and CPU using different random seed
if use_cuda:
    torch.cuda.manual_seed(1)
domain_num = len(config.train_file_list)
max_sent_len = config.max_sent_len

dom2corpus = []
for i in range(domain_num):
    corpus = {}
    corpus['train'] = utils.load_entity_and_relation_sequences(config.train_file_list[i], sep="\t", schema=config.schema)
    corpus['dev'] = utils.load_entity_and_relation_sequences(config.dev_file_list[i], sep="\t", schema=config.schema)
    corpus['test'] = utils.load_entity_and_relation_sequences(config.test_file_list[i], sep="\t", schema=config.schema)

    corpus['train'] = [e for e in corpus['train'] if len(e[0]) <= max_sent_len]
    corpus['dev'] = [e for e in corpus['dev'] if len(e[0]) <= max_sent_len]
    corpus['test'] = [e for e in corpus['test'] if len(e[0]) <= max_sent_len]

    corpus['word_vocab'] = vocab.Vocab("words", PAD="<PAD>", lower=True)
    corpus['char_vocab'] = vocab.Vocab("chars", PAD="<p>", lower=False)
    corpus['ent_span_vocab'] = vocab.Vocab("ent_spans", lower=False)
    corpus['chunk_vocab'] = vocab.Vocab("chunk_tags", lower=False)
    corpus['rel_vocab'] = vocab.Vocab("rel_tags", PAD="None", lower=False)

    utils.create_vocab(corpus['train'] + corpus['dev'] + corpus['test'],
                       [corpus['word_vocab'], corpus['ent_span_vocab'], corpus['chunk_vocab']],
                       corpus['char_vocab'],
                       corpus['rel_vocab'])
    print("Domain %d:" % i)
    print("Total items in train corpus: %s" % len(corpus['train']))
    print("Total items in dev corpus: %s" % len(corpus['dev']))
    print("Total items in test corpus: %s" % len(corpus['test']))
    print("Max sentence length: %s" % max_sent_len)
    dom2corpus.append(corpus)

def get_all_corpus(dom2corpus):
    all_corpus = []
    for corpus in dom2corpus:
        all_corpus.extend(corpus['train'])
        all_corpus.extend(corpus['dev'])
        all_corpus.extend(corpus['test'])
    return all_corpus

word_vocab = vocab.Vocab("words", PAD="<PAD>", lower=True)
char_vocab = vocab.Vocab("chars", PAD="<p>", lower=False)
ent_span_vocab = vocab.Vocab("ent_spans", lower=False)
chunk_vocab = vocab.Vocab("chunk_tags", lower=False)
rel_vocab = vocab.Vocab("rel_tags", PAD="None", lower=False)

all_corpus = get_all_corpus(dom2corpus)

utils.create_vocab(all_corpus,
                   [word_vocab, ent_span_vocab, chunk_vocab],
                   char_vocab,
                   rel_vocab)
for i in range(domain_num):
    dom2corpus[i]['word_vocab'] = word_vocab
    dom2corpus[i]['char_vocab'] = char_vocab
    dom2corpus[i]['ent_span_vocab'] = ent_span_vocab
    dom2corpus[i]['train_tensors'] = utils.data2tensors(dom2corpus[i]['train'],
                                                        [dom2corpus[i]['word_vocab'], dom2corpus[i]['ent_span_vocab'], dom2corpus[i]['chunk_vocab']],
                                                        dom2corpus[i]['rel_vocab'],
                                                        dom2corpus[i]['char_vocab'],
                                                        column_ids=(0, 1, 2))
    dom2corpus[i]['dev_tensors'] = utils.data2tensors(dom2corpus[i]['dev'],
                                                      [dom2corpus[i]['word_vocab'], dom2corpus[i]['ent_span_vocab'], dom2corpus[i]['chunk_vocab']],
                                                      dom2corpus[i]['rel_vocab'],
                                                      dom2corpus[i]['char_vocab'],
                                                      column_ids=(0, 1, 2))
    dom2corpus[i]['test_tensors'] = utils.data2tensors(dom2corpus[i]['test'],
                                                       [dom2corpus[i]['word_vocab'], dom2corpus[i]['ent_span_vocab'], dom2corpus[i]['chunk_vocab']],
                                                       dom2corpus[i]['rel_vocab'],
                                                       dom2corpus[i]['char_vocab'],
                                                       column_ids=(0, 1, 2))
char_embed_kwargs = {
        "vocab_size" : char_vocab.size,
        "embedding_size": config.char_dims,
        "out_channels" : config.char_output_channels,
        "kernel_sizes" : config.char_kernel_sizes
    }

word_char_embedding = WordCharEmbedding(
        word_vocab.size, config.word_dims, char_embed_kwargs,
        dropout=config.dropout, concat=True
    )
word_char_emb_dim = config.word_dims + config.char_output_channels * len(config.char_kernel_sizes)

ent_kwargs_list = []
rel_kwargs_list = []
for i in range(domain_num):
    ent_kwargs = {}
    ent_kwargs['hidden_size'] = config.lstm_hiddens * 2
    ent_kwargs['tag_size'] = dom2corpus[i]['chunk_vocab'].size
    ent_kwargs['use_cuda'] = config.use_cuda
    ent_kwargs_list.append(ent_kwargs)

    rel_kwargs = {}
    rel_kwargs['hidden_size'] = config.lstm_hiddens * 2
    rel_kwargs['chunk_vocab'] = dom2corpus[i]['chunk_vocab']
    rel_kwargs['N_ID'] = dom2corpus[i]['rel_vocab'].PAD_ID
    rel_kwargs['out_channels'] = config.rel_output_channels
    rel_kwargs['kernel_sizes'] = config.rel_kernel_sizes
    rel_kwargs['rel_size'] = dom2corpus[i]['rel_vocab'].size
    rel_kwargs['max_sent_len'] = max_sent_len
    rel_kwargs['use_cuda'] = use_cuda
    rel_kwargs['dropout'] = config.dropout
    rel_kwargs_list.append(rel_kwargs)

ent_span_kwargs = {
    "hidden_size": config.lstm_hiddens,
    "tag_size": ent_span_vocab.size,
    "use_cuda": config.use_cuda
    }
rel_bin_kwargs = {
    "hidden_size": config.lstm_hiddens,
    "chunk_vocab": ent_span_vocab,
    "N_ID": 0,
    "out_channels": config.rel_output_channels,
    "kernel_sizes": config.rel_kernel_sizes,
    "rel_size": 2,
    "max_sent_len": max_sent_len,
    "use_cuda": config.use_cuda,
    "dropout": config.dropout,
    "is_bin_rel": True
    }


mymodel = JointModel(word_char_embedding,
                     word_char_emb_dim,
                     config.lstm_hiddens,
                     ent_kwargs_list,
                     rel_kwargs_list,
                     ent_span_kwargs,
                     rel_bin_kwargs,
                     num_layers=config.lstm_layers,
                     use_cuda=use_cuda,
                     bidirectional=True,
                     sch_k=config.schedule_k,
                     add_share_loss=config.add_share_loss,
                     add_trans_loss=config.add_trans_loss,
                     dropout=config.dropout)
if use_cuda:
    mymodel.cuda()


def predict_all(tensors, batch_size, dom_id):
    mymodel.eval()
    predictions = []
    new_tensors = []
    for i in range(0, len(tensors), batch_size):
        print("[ %d / %d ]" % (len(tensors), min(len(tensors), i + batch_size)))
        batch = tensors[i: i + batch_size]
        X, X_char, Y_span, Y, Y_rel, X_len, batch = utils.get_minibatch(batch, word_vocab, char_vocab)
        X = utils.convert_long_variable(X, use_cuda)
        X_char = utils.convert_long_variable(X_char, use_cuda)
        Y_span = utils.convert_long_tensor(Y_span, use_cuda)
        Y = utils.convert_long_tensor(Y, use_cuda)
        if config.add_share_loss:
          if config.add_trans_loss:
            (ent_loss, ent_span_loss, rel_loss, rel_bin_loss,
            pred_ent_tags, pred_ent_span_tags, pred_rel_tags, pred_rel_bin_tags,
            candi_rel_num, candi_rel_bin_num, _) = mymodel(X, X_char, X_len, Y_span, Y, Y_rel, dom_id)
          else:
            (ent_loss, ent_span_loss, rel_loss, rel_bin_loss,
            pred_ent_tags, pred_ent_span_tags, pred_rel_tags, pred_rel_bin_tags,
            candi_rel_num, candi_rel_bin_num) = mymodel(X, X_char, X_len, Y_span, Y, Y_rel, dom_id)
        else:
          (ent_loss, rel_loss,
           pred_ent_tags, pred_rel_tags, candi_rel_num) = mymodel(X, X_char, X_len, Y_span, Y, Y_rel, dom_id)
        new_tensors.extend(batch)
        predictions.extend(list(zip(pred_ent_tags, pred_rel_tags)))
    return predictions, new_tensors


batch_size = config.batch_size
for dom_id in config.train_domain_list:
    print("Domain : %d" % dom_id)
    state_dict = torch.load(
            open(config.load_model_path_list[dom_id], 'rb'),
            map_location=lambda storage, loc: storage)
    cur_state_dict = mymodel.state_dict()
    for k in state_dict.keys():
      if k in cur_state_dict:
        cur_state_dict[k] = state_dict[k]
    mymodel.load_state_dict(cur_state_dict)
    print("loading previous model successful [%s]" % config.load_model_path_list[dom_id])

    #  print(ent_span_vocab.item2idx)
    #  print(dom2corpus[dom_id]['chunk_vocab'].item2idx)
    #  print(mymodel.ent2span[1].weight)
    #  print(mymodel.ent2span[1].bias)

    for title, tensors in zip( ["train", "dev", "test"],
                               [dom2corpus[dom_id]['train_tensors'],
                                dom2corpus[dom_id]['dev_tensors'],
                                dom2corpus[dom_id]['test_tensors']]):
        if title == "train": continue
        print("\nEvaluating %s" % title)
        predictions, new_tensors = predict_all(tensors, config.batch_size, dom_id)
        eval_path = os.path.join(config.save_dir, "final.%s.output.Domain_%d" % (title, dom_id))
        utils.print_predictions(new_tensors,
                                predictions,
                                eval_path,
                                word_vocab,
                                dom2corpus[dom_id]['chunk_vocab'],
                                dom2corpus[dom_id]['rel_vocab'])
        eval_file(eval_path)
