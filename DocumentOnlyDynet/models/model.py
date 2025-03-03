import sys
import os
import csv
import random

import pickle
import scipy
import pandas as pd
import numpy as np

seed = 10

import dynet_config

# Declare GPU as the default device type
dynet_config.set_gpu()
# Set some parameters manualy
dynet_config.set(mem=400, random_seed=seed)
# Initialize dynet import using above configuration in the current scope
import dynet as dy


from utils.io_utils import IOUtils
from utils.nlp_utils import NLPUtils

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold

random.seed(seed)
np.random.seed(seed)


class Data:
    def __init__(self):
        self.w2i = None
        self.entries = None
        self.train_entries = None
        self.test_entries = None
        self.ext_embedding = None
        self.reviews = None
        self.predicted_reviews = None

    def to(self, device):
        if self.entries:
            for document in self.entries:
                for i in range(len(document.index_tensors)):
                    document.index_tensors[i] = document.index_tensors[i].to(
                        device=device
                    )
        if self.reviews:
            for doc_id in self.reviews:
                for review in self.reviews[doc_id]:
                    review.index_tensor = review.index_tensor.to(device=device)
        if self.predicted_reviews:
            for doc_id in self.predicted_reviews:
                for review in self.predicted_reviews[doc_id]:
                    review.index_tensor = review.index_tensor.to(device=device)

    def load(self, infile):
        with open(infile, "rb") as target:
            self.ext_embeddings, self.entries, self.w2i = pickle.load(target)

    def save_data(self, infile):
        with open(infile, "rb") as target:
            self.ext_embeddings, self.entries, self.w2i = pickle.dump(target)

    def load_predicted_reviews(self, infile):
        with open(infile, "rb") as target:
            self.predicted_reviews = pickle.load(target)
        for app_id in self.predicted_reviews.keys():
            self.predicted_reviews[app_id].sort(
                key=lambda x: x.prediction_result.item(), reverse=True
            )

    def load_reviews(self, infile):
        with open(infile, "rb") as target:
            self.reviews = pickle.load(target)


class Model:
    def __init__(self, data, opt):
        self.opt = opt
        self.model = dy.ParameterCollection()
        self.trainer = dy.MomentumSGDTrainer(self.model)
        self.w2i = data.w2i
        self.wdims = opt.embedding_size
        self.ldims = opt.hidden_size
        self.attsize = opt.attention_size

        self.ext_embeddings = data.ext_embeddings
        # Model Parameters
        self.wlookup = self.model.add_lookup_parameters((len(self.w2i), self.wdims))

        self.__load_external_embeddings()

        if self.opt.encoder_dir == "single":
            #sentence encoder 
            if self.opt.encoder_type == "lstm":
                self.sentence_rnn = [
                    dy.VanillaLSTMBuilder(1, self.wdims, self.ldims, self.model)
                ]
            elif self.opt.encoder_type == "gru":
                self.sentence_rnn = [
                    dy.GRUBuilder(1, self.wdims, self.ldims, self.model)
                ]

            #document encoder
            if self.opt.encoder_type == "lstm":
                self.document_rnn = [
                    dy.VanillaLSTMBuilder(1, self.ldims + 2 * self.ldims, self.ldims, self.model)
                ]
            elif self.opt.encoder_type == "gru":
                self.document_rnn = [
                    dy.GRUBuilder(1, self.ldims + 2 * self.ldims, self.ldims, self.model)
                ]

            #classifier
            self.mlp_w = self.model.add_parameters((1, self.ldims + 2 * self.ldims))
            self.mlp_b = self.model.add_parameters(1)
        elif self.opt.encoder_dir == "bidirectional":
            #sentence encoder 
            if self.opt.encoder_type == "lstm":
                self.sentence_rnn = [
                    dy.VanillaLSTMBuilder(1, self.wdims, self.ldims, self.model),
                    dy.VanillaLSTMBuilder(1, self.wdims, self.ldims, self.model),
                ]
            elif self.opt.encoder_type == "gru":
                self.sentence_rnn = [
                    dy.GRUBuilder(1, self.wdims, self.ldims, self.model),
                    dy.GRUBuilder(1, self.wdims, self.ldims, self.model),
                ]

            #document encoder
            if self.opt.encoder_type == "lstm":
                self.document_rnn = [
                    dy.VanillaLSTMBuilder(1, 2 * self.ldims + 4 * self.ldims, self.ldims, self.model),
                    dy.VanillaLSTMBuilder(1, 2 * self.ldims + 4 * self.ldims, self.ldims, self.model)
                ]
            elif self.opt.encoder_type == "gru":
                self.document_rnn = [
                    dy.GRUBuilder(1, 2 * self.ldims + 4 * self.ldims, self.ldims, self.model),
                    dy.GRUBuilder(1, 2 * self.ldims + 4 * self.ldims, self.ldims, self.model)
                ]
       
            #classifier
            self.mlp_w = self.model.add_parameters((1, 2 * self.ldims + 4 * self.ldims))
            self.mlp_b = self.model.add_parameters(1)

    def __load_external_embeddings(self):
        print("Initializing word embeddings by pre-trained vectors")
        count = 0
        for word in self.w2i:
            if word in self.ext_embeddings:
                count += 1
                self.wlookup.init_row(self.w2i[word], self.ext_embeddings[word])
        print(
            "Vocab size: %d; #words having pretrained vectors: %d"
            % (len(self.w2i), count)
        )


def write_file(filename, string):
    with open(filename, "a") as target:
        target.write("{}\n".format(string))
        target.flush()


def encode_sequence(model, seq, rnn_builder):
    def predict_sequence(builder, inputs):
        s_init = builder.initial_state()
        return s_init.transduce(inputs)

    if model.opt.encoder_dir == "bidirectional":
        f_in = [entry for entry in seq]
        b_in = [rentry for rentry in reversed(seq)]
        forward_sequence = predict_sequence(rnn_builder[0], f_in)
        backward_sequence = predict_sequence(rnn_builder[1], b_in)
        return [
            dy.concatenate([s1, s2])
            for s1, s2 in zip(forward_sequence, backward_sequence)
        ]
    elif model.opt.encoder_dir == "single":
        f_in = [entry for entry in seq]
        state = rnn_builder[0].initial_state()
        states = []
        for entry in seq:
            state = state.add_input(entry)
            states.append(state.output())
        return states


def max_pooling(encoded_sequence):
    values = np.array([encoding.value() for encoding in encoded_sequence])
    min_indexes = np.argmax(values, axis=0)
    pooled_context = dy.concatenate(
        [encoded_sequence[row][col] for col, row in enumerate(min_indexes)]
    )
    return pooled_context


def min_pooling(encoded_sequence):
    values = np.array([encoding.value() for encoding in encoded_sequence])
    min_indexes = np.argmin(values, axis=0)
    pooled_context = dy.concatenate(
        [encoded_sequence[row][col] for col, row in enumerate(min_indexes)]
    )
    return pooled_context


def average_pooling(encoded_sequence):
    averages = []
    for col in range(encoded_sequence[0].dim()[0][0]):
        avg = []
        for row in range(len(encoded_sequence)):
            avg.append(encoded_sequence[row][col])
        averages.append(dy.average(avg))
    return dy.concatenate(averages)


def train_item(args, model, document):
    loss = None
    word_lookups = []
    for preprocessed_sentence in document.preprocessed_sentences:
        seq = [
            model.wlookup[int(model.w2i.get(entry, 0))]
            for entry in preprocessed_sentence
        ]
        if len(seq) > 0:
            word_lookups.append(seq)

    sentences_lookups = []
    for seq in word_lookups:
        sentence_encode = encode_sequence(model, seq, model.sentence_rnn)
        global_max = max_pooling(sentence_encode)
        global_min = average_pooling(sentence_encode)
        if len(sentence_encode) > 0:
            last_out = sentence_encode[-1]
            context = dy.concatenate([last_out, global_max, global_min])
            sentences_lookups.append(context)

    document_encode = encode_sequence(model, sentences_lookups, model.document_rnn)
    global_max = max_pooling(document_encode)
    global_min = average_pooling(document_encode)
    if len(document_encode) > 0:
        last_out = sentence_encode[-1]
        context = dy.concatenate([last_out, global_max, global_min])
        y_pred = dy.logistic((model.mlp_w * context) + model.mlp_b)

        if document.permissions[args.permission_type]:
            loss = dy.binary_log_loss(y_pred, dy.scalarInput(1))
        else:
            loss = dy.binary_log_loss(y_pred, dy.scalarInput(0))

        loss.backward()
        model.trainer.update()
        loss_val = loss.scalar_value()
        dy.renew_cg()
        return loss_val
    return 0


def test_item(model, document):
    word_lookups = []
    for preprocessed_sentence in document.preprocessed_sentences:
        seq = [
            model.wlookup[int(model.w2i.get(entry, 0))]
            for entry in preprocessed_sentence
        ]
        if len(seq) > 0:
            word_lookups.append(seq)

    sentences_lookups = []
    for seq in word_lookups:
        sentence_encode = encode_sequence(model, seq, model.sentence_rnn)
        global_max = max_pooling(sentence_encode)
        global_min = average_pooling(sentence_encode)
        if len(sentence_encode) > 0:
            last_out = sentence_encode[-1]
            context = dy.concatenate([last_out, global_max, global_min])
            sentences_lookups.append(context)

    document_encode = encode_sequence(model, sentences_lookups, model.document_rnn)
    global_max = max_pooling(document_encode)
    global_min = average_pooling(document_encode)
    if len(document_encode) > 0:
        last_out = sentence_encode[-1]
        context = dy.concatenate([last_out, global_max, global_min])
        y_pred = dy.logistic((model.mlp_w * context) + model.mlp_b)
        document.prediction_result = y_pred.scalar_value()
        dy.renew_cg()
        return document.prediction_result
    return 0


def train_all(args, model, data):
    write_file(args.outdir, "Training...")
    losses = []
    for index, document in enumerate(data.train_entries):
        loss = train_item(args, model, document)
        if index != 0:
            if index % model.opt.print_every == 0:
                write_file(
                    args.outdir,
                    "Index {} Loss {}".format(
                        index, np.mean(losses[index - model.opt.print_every :])
                    ),
                )
        losses.append(loss)


def test_all(args, model, data):
    def pr_roc_auc(predictions, gold):
        y_true = np.array(gold)
        y_scores = np.array(predictions)
        roc_auc = roc_auc_score(y_true, y_scores)
        pr_auc = average_precision_score(y_true, y_scores)
        return roc_auc, pr_auc

    write_file(args.outdir, "Predicting..")

    predictions, gold = [], []
    for index, document in enumerate(data.test_entries):
        pred = test_item(model, document)
        predictions.append(pred)
        gold.append(document.permissions[args.permission_type])
    return pr_roc_auc(predictions, gold)


def kfold_validation(args, data):
    data.entries = np.array(data.entries)
    random.shuffle(data.entries)

    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    roc_l, pr_l = [], []
    for foldid, (train, test) in enumerate(kfold.split(data.entries)):
        write_file(args.outdir, "Fold {}".format(foldid + 1))

        model = Model(data, args)
        data.train_entries = data.entries[train]
        data.test_entries = data.entries[test]
        max_roc_auc, max_pr_auc = 0, 0
        for epoch in range(args.num_epoch):
            train_all(args, model, data)
            roc_auc, pr_auc = test_all(args, model, data)
            if pr_auc > max_pr_auc:
                max_pr_auc = pr_auc
                max_roc_auc = roc_auc
            write_file(
                args.outdir, "Epoch {} ROC {}  PR {}".format(epoch + 1, roc_auc, pr_auc)
            )

        write_file(args.outdir, "ROC {} PR {}".format(max_roc_auc, max_pr_auc))
        roc_l.append(max_roc_auc)
        pr_l.append(max_pr_auc)
    write_file(
        args.outdir, "Summary : ROC {} PR {}".format(np.mean(roc_l), np.mean(pr_l))
    )


def run(args):
    data = Data()
    data.load(args.saved_data)

    kfold_validation(args, data)
