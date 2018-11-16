#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 17/06/05 20:57:52

@author: Changzhi Sun
"""
import os
import json
import re
import numpy as np

import torch
from torch.autograd import Variable


def load_entity_and_relation_sequences(filenames, sep="\t", schema="BIO"):
    def convert_sequence(source_path, target_path):
        fsource = open(source_path, "r", encoding="utf8")
        ftarget = open(target_path, "w", encoding="utf8")
        for line in fsource:
            sent = json.loads(line)
            tokens = sent['sentText'].split(' ')
            tags = ['O'] * len(tokens)
            id2ent = {}
            for men in sent['entityMentions']:
                id2ent[men['emId']] = men['offset']
                s, e = men['offset']
                if schema == "BIO":
                    tags[s] = 'B-' + men['label']
                    for j in range(s+1, e):
                        tags[j] = 'I-' + men['label']
                else:
                    if e - s == 1:
                        tags[s] = "U-" + men['label']
                    elif e - s == 2:
                        tags[s] = 'B-' + men['label']
                        tags[s+1] = 'E-' + men['label']
                    else:
                        tags[s] = 'B-' + men['label']
                        tags[e - 1] = 'E-' + men['label']
                        for j in range(s+1, e - 1):
                            tags[j] = 'I-' + men['label']
            for w, t in zip(tokens, tags):
                if t != "O":
                    print("{0}{1}{2}{3}{4}".format(w, sep, t[0]+"-ENT", sep, t), file=ftarget)
                else:
                    print("{0}{1}{2}{3}{4}".format(w, sep, t, sep, t), file=ftarget)
            for men in sent['relationMentions']:
                em1_idx = id2ent[men['em1Id']]
                em2_idx = id2ent[men['em2Id']]
                em1_text = men['em1Text']
                em2_text = men['em2Text']
                direction = "-->"
                if em1_idx[0] > em2_idx[0]:
                    direction = "<--"
                    em1_idx, em2_idx = em2_idx, em1_idx
                    em1_text, em2_text = em2_text, em1_text
                label = men['label'] + direction
                print("{0}\t{1}\t{2}\t{3}\t{4}".format(
                    em1_idx, em2_idx,
                    em1_text, em2_text,
                    label), file=ftarget)
            print(file=ftarget)
        fsource.close()
        ftarget.close()

    sequences = []
    if isinstance(filenames, str):
        filenames = [filenames]
    for filename in filenames:
        ent_rel_filename = "entity_relation.txt"
        target_filename = os.path.join(os.path.dirname(filename), ent_rel_filename)
        convert_sequence(filename, target_filename)
        with open(target_filename, "r", encoding='utf-8') as fp:
            seq = [[], []]
            for line in fp:
                line = line.rstrip()
                if line:
                    line = line.split(sep)
                    line = [line[idx] for idx in range(len(line))]
                    if len(line) == 3:
                        seq[0].append(tuple(line))
                    elif len(line) == 5:
                        seq[1].append((eval(line[0]), eval(line[1]), line[-1]))
                else:
                    if seq[0]:
                        sequences.append(seq)
                    seq = [[], []]
            if seq[0]:
                sequences.append(seq)
    return sequences


def load_word_vectors(vector_file, ndims, vocab):
    #  W = np.zeros((vocab.size, ndims), dtype="float32")
    W = np.random.uniform(-0.25, 0.25, (vocab.size, ndims))
    total, found = 0, 0
    with open(vector_file) as fp:
        for i, line in enumerate(fp):
            line = line.rstrip().split()
            if line:
                total += 1
                try:
                    assert len(line) == ndims+1,(
                        "Line[{}] {} vector dims {} doesn't match ndims={}".format(i, line[0], len(line)-1, ndims)
                    )
                except AssertionError as e:
                    print(e)
                    continue
                word = line[0]
                idx = vocab.getidx(word)
                if idx >= vocab.offset:
                    found += 1
                    vecs = np.array(list(map(float, line[1:])))
                    W[idx, :] = vecs
    # Write to cache file
    print("Found {} [{:.2f}%] vectors from {} vectors in {} with ndims={}".format(
        found, found * 100/vocab.size, total, vector_file, ndims))
    #  norm_W = np.sqrt((W*W).sum(axis=1, keepdims=True))
    #  valid_idx = norm_W.squeeze() != 0
    #  W[valid_idx, :] /= norm_W[valid_idx]
    return W


def get_minibatch(batch, word_vocab, char_vocab):
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    X_len = [len(s[0]) for s in batch]
    max_batch_sent_len = max(X_len)
    max_batch_char_len = max([len(c) for s in batch for c in s[-2]])

    X = []
    X_char = []
    Y_span = []
    Y = []
    Y_rel = []
    for s in batch:
        X.append(s[0] + [word_vocab.PAD_ID] * (max_batch_sent_len - len(s[0])))
        char_pad = []
        for c in s[-2]:
            char_pad.append(c + [char_vocab.PAD_ID] * (max_batch_char_len - len(c)))
        X_char.append(char_pad + [[char_vocab.PAD_ID] * max_batch_char_len] * (max_batch_sent_len - len(s[0])))
        Y_span.append(s[1] + [-1] * (max_batch_sent_len - len(s[1])))
        Y.append(s[2] + [-1] * (max_batch_sent_len - len(s[1])))
        Y_rel.append(s[-1])
    X = np.array(X)
    X_char = np.array(X_char)
    Y_span = np.array(Y_span)
    Y = np.array(Y)
    return X, X_char, Y_span, Y, Y_rel, X_len, batch


def create_vocab(data, vocabs, char_vocab, rel_vocab, word_idx=0):
    n_vocabs = len(vocabs)
    for sent in data:
        for token_tags in sent[0]:
            for vocab_id in range(n_vocabs):
                vocabs[vocab_id].add(token_tags[vocab_id])
            char_vocab.batch_add(token_tags[word_idx])
        for rel_tags in sent[1]:
            rel_vocab.add(rel_tags[-1])
    print("Created vocabs: %s, relation[%s], chars[%s]" % (", ".join(
        "{}[{}]".format(vocab.name, vocab.size)
        for vocab in vocabs
    ), rel_vocab.size, char_vocab.size))


def data2tensors(data, vocabs, rel_vocab, char_vocab, word_idx=0, column_ids=(0, -1)):
    vocabs = [vocabs[idx] for idx in column_ids]
    n_vocabs = len(vocabs)
    tensors = []
    for sent in data:
        sent_vecs = [[] for i in range(n_vocabs+2)] # Last two are for char and relation vecs
        char_vecs = []
        for token_tags in sent[0]:
            vocab_id = 0 # First column is the word
            # lowercase the word
            sent_vecs[vocab_id].append(
                    vocabs[vocab_id].getidx(token_tags[vocab_id].lower())
                )
            for vocab_id in range(1, n_vocabs):
                sent_vecs[vocab_id].append(
                    vocabs[vocab_id].getidx(token_tags[vocab_id])
                )
            sent_vecs[-2].append(
                [char_vocab.getidx(c) for c in token_tags[word_idx]]
            )
        for b, e, t in sent[1]:
            sent_vecs[-1].append([b, e, rel_vocab.getidx(t)])
        tensors.append(sent_vecs)
    return tensors

def print_predictions(corpus,
                      predictions,
                      filename,
                      word_vocab,
                      chunk_vocab,
                      rel_vocab):
    with open(filename, "w", encoding="utf8") as fp:
        i = 0
        for seq, pred in zip(corpus, predictions):
            i += 1
            #  print(i, pred)
            seq_len = len(seq[0])
            assert len(seq[0]) == len(pred[0])
            assert len(seq[1]) == len(pred[0])
            for (idx, true_label), pred_label in zip(zip(seq[0], seq[2]), pred[0]):
                pred_label = chunk_vocab.idx2item[pred_label]
                token = word_vocab.idx2item[idx]
                true_label = chunk_vocab.idx2item[true_label]
                print("{}\t{}\t{}".format(token, true_label, pred_label), file=fp)
            for s, e, r in seq[-1]:
                r = rel_vocab.idx2item[r]
                if r[-3:] == "<--":
                    s, e = e, s
                r = r[:-3]
                print("Rel-True\t{}\t{}\t{}".format(s, e, r), file=fp)
            for s, e, r in pred[1]:
                r = rel_vocab.idx2item[r]
                assert int(s[-1]) < len(pred[0])
                assert int(e[-1]) < len(pred[0])
                s = [s[0], s[-1] + 1]
                e = [e[0], e[-1] + 1]
                if r[-3:] == "<--":
                    s, e = e, s
                r = r[:-3]
                print("Rel-Pred\t{}\t{}\t{}".format(s, e, r), file=fp)
            print(file=fp) # Add new line after each sequence


def print_share_predictions(corpus,
                            predictions,
                            filename,
                            word_vocab,
                            chunk_vocab):
    with open(filename, "w", encoding="utf8") as fp:
        i = 0
        for seq, pred in zip(corpus, predictions):
            i += 1
            #  print(i, pred)
            seq_len = len(seq[0])
            assert len(seq[0]) == len(pred[0])
            assert len(seq[1]) == len(pred[0])
            for (idx, true_label), pred_label in zip(zip(seq[0], seq[1]), pred[0]):
                pred_label = chunk_vocab.idx2item[pred_label]
                token = word_vocab.idx2item[idx]
                true_label = chunk_vocab.idx2item[true_label]
                print("{}\t{}\t{}".format(token, true_label, pred_label), file=fp)
            for s, e, r in seq[-1]:
                r = "YES"
                print("Rel-True\t{}\t{}\t{}".format(s, e, r), file=fp)
            for s, e, r in pred[1]:
                s = [s[0], s[-1] + 1]
                e = [e[0], e[-1] + 1]
                r = "YES"
                print("Rel-Pred\t{}\t{}\t{}".format(s, e, r), file=fp)
            print(file=fp) # Add new line after each sequence


def convert_long_tensor(var, use_cuda):
    var = torch.LongTensor(var)
    if use_cuda:
        var = var.cuda(async=True)
    return var


def convert_float_tensor(var, use_cuda):
    var = torch.FloatTensor(var)
    if use_cuda:
        var = var.cuda(async=True)
    return var


def convert_long_variable(var, use_cuda):
    return Variable(convert_long_tensor(var, use_cuda))


def convert_float_variable(var, use_cuda):
    return Variable(convert_float_tensor(var, use_cuda))


def assign_embeddings(embedding_module, pretrained_embeddings, fix_embedding=False):
    embedding_module.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
    if fix_embedding:
        embedding_module.weight.requires_grad = False


def parse_tag(t):
    m = re.match(r'^([^-]*)-(.*)$', t)
    return m.groups() if m else (t, '')


def end_of_chunk(prev_tag, tag, prev_type, type_):
    # check if a chunk ended between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'U': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'U': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'U': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    # these chunks are assumed to have length 1
    if prev_tag == ']': chunk_end = True
    if prev_tag == '[': chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    # check if a chunk started between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'U': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'U' and tag == 'E': chunk_start = True
    if prev_tag == 'U' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    # these chunks are assumed to have length 1
    if tag == '[': chunk_start = True
    if tag == ']': chunk_start = True

    return chunk_start
