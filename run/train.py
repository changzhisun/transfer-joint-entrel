#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 17/06/05 21:15:15

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


pretrained_embeddings = utils.load_word_vectors(config.pretrained_embeddings_file,
                                                config.word_dims,
                                                word_vocab)
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
utils.assign_embeddings(word_char_embedding.word_embeddings, pretrained_embeddings)
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

if os.path.exists(config.load_model_path_list[1]):
    state_dict = torch.load(
            open(config.load_model_path_list[1], 'rb'),
            map_location=lambda storage, loc: storage)
    mymodel.load_state_dict(state_dict)
    print("loading previous model successful [%s]" % config.load_model_path_list[1])

def get_fixed_matrix(span_vocab, chunk_vocab):
    span_size = span_vocab.size
    chunk_size = chunk_vocab.size
    mat = np.zeros((span_size, chunk_size))
    for i in range(span_size):
        for j in range(chunk_size):
            if span_vocab.idx2item[i] == 'O' and chunk_vocab.idx2item[j] == 'O':
                mat[i][j] = 1.0
            elif span_vocab.idx2item[i].startswith('B') and chunk_vocab.idx2item[j].startswith('B'):
                mat[i][j] = 1.0
            elif span_vocab.idx2item[i].startswith('I') and chunk_vocab.idx2item[j].startswith('I'):
                mat[i][j] = 1.0
            elif span_vocab.idx2item[i].startswith('E') and chunk_vocab.idx2item[j].startswith('E'):
                mat[i][j] = 1.0
            elif span_vocab.idx2item[i].startswith('U') and chunk_vocab.idx2item[j].startswith('U'):
                mat[i][j] = 1.0
    return mat

#  fixed_mat_0 = get_fixed_matrix(ent_span_vocab, dom2corpus[0]['chunk_vocab'])
#  fixed_mat_1 = get_fixed_matrix(ent_span_vocab, dom2corpus[1]['chunk_vocab'])
#  utils.assign_embeddings(mymodel.ent2span[0], fixed_mat_0, fix_embedding=True)
#  utils.assign_embeddings(mymodel.ent2span[1], fixed_mat_1, fix_embedding=True)

parameters = [p for p in mymodel.parameters() if p.requires_grad]
optimizer = optim.Adadelta(parameters)

def step(batch, dom_id, i_epoch=0):
    X, X_char, Y_span, Y, Y_rel, X_len, batch = utils.get_minibatch(batch, word_vocab, char_vocab)
    X = utils.convert_long_variable(X, use_cuda)
    X_char = utils.convert_long_variable(X_char, use_cuda)
    Y_span = utils.convert_long_tensor(Y_span, use_cuda)
    Y = utils.convert_long_tensor(Y, use_cuda)
    if config.add_share_loss:
        if config.add_trans_loss:
            (ent_loss, ent_span_loss, rel_loss, rel_bin_loss,
            pred_ent_tags, pred_ent_span_tags, pred_rel_tags, pred_rel_bin_tags,
            candi_rel_num, candi_rel_bin_num, trans_loss) = mymodel(X, X_char, X_len, Y_span, Y, Y_rel, dom_id, i_epoch)
            return (ent_loss, ent_span_loss, rel_loss, rel_bin_loss, trans_loss,
                    pred_ent_tags, pred_ent_span_tags, pred_rel_tags, pred_rel_bin_tags,
                    candi_rel_num, candi_rel_bin_num, X_len, batch)
        (ent_loss, ent_span_loss, rel_loss, rel_bin_loss,
        pred_ent_tags, pred_ent_span_tags, pred_rel_tags, pred_rel_bin_tags,
        candi_rel_num, candi_rel_bin_num) = mymodel(X, X_char, X_len, Y_span, Y, Y_rel, dom_id, i_epoch)
        return (ent_loss, ent_span_loss, rel_loss, rel_bin_loss,
                pred_ent_tags, pred_ent_span_tags, pred_rel_tags, pred_rel_bin_tags,
                candi_rel_num, candi_rel_bin_num, X_len, batch)
    else:
        (ent_loss, rel_loss, pred_ent_tags,
         pred_rel_tags, candi_rel_num)  = mymodel(X, X_char, X_len, Y_span, Y, Y_rel, dom_id, i_epoch)
        return (ent_loss, rel_loss, 
                pred_ent_tags, pred_rel_tags, 
                candi_rel_num, X_len, batch)


def train_step(batch, optimizer, dom_id, i_epoch):
    optimizer.zero_grad()
    mymodel.train()
    if config.add_share_loss:
        if config.add_trans_loss:
            (ent_loss, ent_span_loss, rel_loss, rel_bin_loss, trans_loss,
            pred_ent_tags, pred_ent_span_tags, pred_rel_tags, pred_rel_bin_tags,
            candi_rel_num, candi_rel_bin_num, X_len, batch) = step(batch, dom_id, i_epoch)
        else:
            (ent_loss, ent_span_loss, rel_loss, rel_bin_loss,
            pred_ent_tags, pred_ent_span_tags, pred_rel_tags, pred_rel_bin_tags,
            candi_rel_num, candi_rel_bin_num, X_len, batch) = step(batch, dom_id, i_epoch)
        if candi_rel_bin_num == 0:
            rel_bin_loss = utils.convert_float_variable([0], use_cuda)
            rel_bin_loss.requires_grad = True
        else:
            rel_bin_loss = rel_bin_loss / candi_rel_bin_num
        ent_span_loss = ent_span_loss / sum(X_len)
    else:
        (ent_loss, rel_loss, pred_ent_tags, pred_rel_tags, 
         candi_rel_num,  X_len, batch) = step(batch, dom_id, i_epoch)
    if candi_rel_num == 0:
        rel_loss = utils.convert_float_variable([0], use_cuda)
        rel_loss.requires_grad = True
    else:
        rel_loss = rel_loss / candi_rel_num

    ent_loss = ent_loss / sum(X_len)
    if config.add_share_loss:
        if config.add_trans_loss:
            loss = ent_loss + rel_loss + ent_span_loss + rel_bin_loss + trans_loss
        else:
            loss = ent_loss + rel_loss + ent_span_loss + rel_bin_loss
    else:
        loss = ent_loss + rel_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters, config.clip_c)
    optimizer.step()
    if config.add_share_loss:
        if config.add_trans_loss:
            print('Domain : %d Epoch : %d Minibatch : %d Loss : %.5f\t(%.5f, %.5f, %.5f, %.5f, %.5f)' % (
                dom_id, i_epoch, i,
                loss.item(),
                ent_loss.item(), rel_loss.item(),
                ent_span_loss.item(), rel_bin_loss.item(),
                trans_loss.item()))
        else:
            print('Domain : %d Epoch : %d Minibatch : %d Loss : %.5f\t(%.5f, %.5f, %.5f, %.5f)' % (
                dom_id, i_epoch, i,
                loss.item(),
                ent_loss.item(), rel_loss.item(),
                ent_span_loss.item(), rel_bin_loss.item()))
    else:
        print('Domain : %d Epoch : %d Minibatch : %d Loss : %.5f\t(%.5f, %.5f)' % (
            dom_id, i_epoch, i,
            loss.item(),
            ent_loss.item(), rel_loss.item()))


def dev_step(dev_tensors, batch_size, dom_id, i_epoch):
    optimizer.zero_grad()
    mymodel.eval()
    predictions = []
    ent_losses = []
    rel_losses = []
    if config.add_share_loss:
        ent_span_losses = []
        rel_bin_losses = []
        share_predictions = []
        if config.add_trans_loss:
            trans_losses = []
    new_tensors = []
    all_ent_num = 0
    all_rel_num = 0
    all_rel_bin_num = 0
    for k in range(0, len(dev_tensors), batch_size):
        batch = dev_tensors[k: k + batch_size]
        if config.add_share_loss:
            if config.add_trans_loss:
                (ent_loss, ent_span_loss, rel_loss, rel_bin_loss, trans_loss,
                pred_ent_tags, pred_ent_span_tags, pred_rel_tags, pred_rel_bin_tags,
                candi_rel_num, candi_rel_bin_num, X_len, batch) = step(batch, dom_id, i_epoch)
                trans_losses.append(trans_loss.item())
            else:
                (ent_loss, ent_span_loss, rel_loss, rel_bin_loss,
                pred_ent_tags, pred_ent_span_tags, pred_rel_tags, pred_rel_bin_tags,
                candi_rel_num, candi_rel_bin_num, X_len, batch) = step(batch, dom_id, i_epoch)
            all_rel_bin_num += candi_rel_bin_num
            ent_span_losses.append(ent_span_loss.item())
            rel_bin_losses.append(rel_bin_loss.item())
            share_predictions.extend(list(zip(pred_ent_span_tags, pred_rel_bin_tags)))
        else:
            (ent_loss, rel_loss, pred_ent_tags, pred_rel_tags, 
            candi_rel_num,  X_len, batch) = step(batch, dom_id, i_epoch)
        all_rel_num += candi_rel_num
        all_ent_num += sum(X_len)
        predictions.extend(list(zip(pred_ent_tags, pred_rel_tags)))
        ent_losses.append(ent_loss.item())
        rel_losses.append(rel_loss.item())
        new_tensors.extend(batch)
    ent_loss = sum(ent_losses) / all_ent_num
    if all_rel_num == 0:
        rel_loss = 0
    else:
        rel_loss = sum(rel_losses) / all_rel_num
    if config.add_share_loss:
        ent_span_loss = sum(ent_span_losses) / all_ent_num
        if all_rel_bin_num == 0:
            rel_bin_loss = 0
        else:
            rel_bin_loss = sum(rel_bin_losses) / all_rel_bin_num
        if config.add_trans_loss:
            trans_loss = sum(trans_losses) / len(trans_losses)
            loss = ent_loss + rel_loss + ent_span_loss + rel_bin_loss + trans_loss
            print('Domain : %d Epoch : %d Minibatch : %d Loss : %.5f\t(%.5f, %.5f, %.5f, %.5f, %.5f)' % (
                dom_id, i_epoch, i,
                loss,
                ent_loss, rel_loss,
                ent_span_loss, rel_bin_loss,
                trans_loss))
        else:
            loss = ent_loss + rel_loss + ent_span_loss + rel_bin_loss
            print('Domain : %d Epoch : %d Minibatch : %d Loss : %.5f\t(%.5f, %.5f, %.5f, %.5f)' % (
                dom_id, i_epoch, i,
                loss,
                ent_loss, rel_loss,
                ent_span_loss, rel_bin_loss))
    else:
        loss = ent_loss + rel_loss
        print('Domain : %d Epoch : %d Minibatch : %d Loss : %.5f\t(%.5f, %.5f)' % (
            dom_id, i_epoch, i,
            loss,
            ent_loss, rel_loss))

    eval_path = os.path.join(config.save_dir, "validate.dev.output.Domain_%d" % dom_id)
    share_eval_path = os.path.join(config.save_dir, "validate.dev.output.share.Domain_%d" % dom_id)
    utils.print_predictions(new_tensors,
                            predictions,
                            eval_path,
                            word_vocab,
                            dom2corpus[dom_id]['chunk_vocab'],
                            dom2corpus[dom_id]['rel_vocab'])
    entity_score, relation_score = eval_file(eval_path)
    if config.add_share_loss:
        print("Share Task Evaluation (Dev)...")
        utils.print_share_predictions(new_tensors,
                                      share_predictions,
                                      share_eval_path,
                                      word_vocab,
                                      dom2corpus[dom_id]['ent_span_vocab'])
        eval_file(share_eval_path)
    return relation_score

def get_batch(data, batch_size, i, i_epoch):
    if i == 0:
        np.random.shuffle(data)
        i_epoch += 1
    if i + batch_size >= len(data):
        return data[i : i + batch_size], 0, i_epoch
    return data[i: i + batch_size], i + batch_size, i_epoch

batch_ids = [0 for _ in range(domain_num)]
epochs = [-1 for _ in range(domain_num)]
best_f1 = [0 for _ in range(domain_num)]
batch_size = config.batch_size

for i in range(config.train_iters):

    for dom_id in config.train_domain_list:
        batch, batch_ids[dom_id], epochs[dom_id] = get_batch(dom2corpus[dom_id]['train_tensors'], batch_size, batch_ids[dom_id], epochs[dom_id])
        train_step(batch, optimizer, dom_id, epochs[dom_id])

        if i > 0 and i % config.validate_every == 0:
            print("Domain %d:" % dom_id)
            print('Evaluating model in dev set...')
            dev_f1 = dev_step(dom2corpus[dom_id]['dev_tensors'], batch_size, dom_id, epochs[dom_id])

            if dev_f1 > best_f1[dom_id]:
                best_f1[dom_id] = dev_f1
                print('Saving model for Domain %d...' % dom_id)
                torch.save(mymodel.state_dict(),
                    open(os.path.join(config.save_dir, "minibatch", 'epoch__%d__minibatch_%d.Domain_%d' % (epochs[dom_id], i, dom_id)), "wb"))
                torch.save(mymodel.state_dict(), open(config.save_model_path + ".Domain_%d" % dom_id, "wb"))
print("end")
