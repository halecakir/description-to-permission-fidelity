import sys
import os
import csv
import random

import pickle
import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
        self.ext_embeddings = None
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
            self.ext_embeddings, self.entries, self.w2i, self.reviews = pickle.load(target)

    def save_data(self, infile):
        with open(infile, "wb") as target:
            pickle.dump([self.ext_embeddings, self.entries, self.w2i], target)

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
            
            #word level attention
            self.word_attention_w = self.model.add_parameters((self.attsize, self.ldims))
            self.word_attention_b = self.model.add_parameters(self.attsize)
            self.word_att_context = self.model.add_parameters(self.attsize)
            
            #document encoder
            if self.opt.encoder_type == "lstm":
                self.document_rnn = [
                    dy.VanillaLSTMBuilder(1, self.ldims + 2 * self.ldims, self.ldims, self.model)
                ]
            elif self.opt.encoder_type == "gru":
                self.document_rnn = [
                    dy.GRUBuilder(1, self.ldims + 2 * self.ldims, self.ldims, self.model)
                ]

            #sentence level attention
            self.sentence_attention_w = self.model.add_parameters((self.attsize, self.ldims))
            self.sentence_attention_b = self.model.add_parameters(self.attsize)
            self.sentence_att_context = self.model.add_parameters(self.attsize)  

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

            #word level attention
            self.word_attention_w = self.model.add_parameters((self.attsize, 2 * self.ldims))
            self.word_attention_b = self.model.add_parameters(self.attsize)
            self.word_att_context = self.model.add_parameters(self.attsize)
            
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

            #sentence level attention
            self.sentence_attention_w = self.model.add_parameters((self.attsize, 2 * self.ldims))
            self.sentence_attention_b = self.model.add_parameters(self.attsize)
            self.sentence_att_context = self.model.add_parameters(self.attsize)  
            
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
        
    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.populate(path)

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
            att_mlp_outputs = []
            for e in sentence_encode:
                mlp_out = (model.word_attention_w * e) + model.word_attention_b
                att_mlp_outputs.append(mlp_out)

            lst = []
            for o in att_mlp_outputs:
                lst.append(dy.exp(dy.sum_elems(dy.cmult(o, model.word_att_context))))

            sum_all = dy.esum(lst)

            probs = [dy.cdiv(e, sum_all) for e in lst]
            att_context = dy.esum(
                [dy.cmult(p, h) for p, h in zip(probs, sentence_encode)]
            )
            context = dy.concatenate([att_context, global_max, global_min])
            sentences_lookups.append(context)

    document_encode = encode_sequence(model, sentences_lookups, model.document_rnn)
    global_max = max_pooling(document_encode)
    global_min = average_pooling(document_encode)
    if len(document_encode) > 0:
        att_mlp_outputs = []
        for e in document_encode:
            mlp_out = (model.sentence_attention_w * e) + model.sentence_attention_b
            att_mlp_outputs.append(mlp_out)

        lst = []
        for o in att_mlp_outputs:
            lst.append(dy.exp(dy.sum_elems(dy.cmult(o, model.sentence_att_context))))

        sum_all = dy.esum(lst)

        probs = [dy.cdiv(e, sum_all) for e in lst]
        att_context = dy.esum(
            [dy.cmult(p, h) for p, h in zip(probs, document_encode)]
        )
        context = dy.concatenate([att_context, global_max, global_min])
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


def get_context(model, preprocessed_sentences):
    word_lookups = []
    for preprocessed_sentence in preprocessed_sentences:
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
            att_mlp_outputs = []
            for e in sentence_encode:
                mlp_out = (model.word_attention_w * e) + model.word_attention_b
                att_mlp_outputs.append(mlp_out)

            lst = []
            for o in att_mlp_outputs:
                lst.append(dy.exp(dy.sum_elems(dy.cmult(o, model.word_att_context))))

            sum_all = dy.esum(lst)

            probs = [dy.cdiv(e, sum_all) for e in lst]
            att_context = dy.esum(
                [dy.cmult(p, h) for p, h in zip(probs, sentence_encode)]
            )
            context = dy.concatenate([att_context, global_max, global_min])
            sentences_lookups.append(context)

    document_encode = encode_sequence(model, sentences_lookups, model.document_rnn)
    global_max = max_pooling(document_encode)
    global_min = average_pooling(document_encode)
    if len(document_encode) > 0:
        att_mlp_outputs = []
        for e in document_encode:
            mlp_out = (model.sentence_attention_w * e) + model.sentence_attention_b
            att_mlp_outputs.append(mlp_out)

        lst = []
        for o in att_mlp_outputs:
            lst.append(dy.exp(dy.sum_elems(dy.cmult(o, model.sentence_att_context))))

        sum_all = dy.esum(lst)

        probs = [dy.cdiv(e, sum_all) for e in lst]
        att_context = dy.esum(
            [dy.cmult(p, h) for p, h in zip(probs, document_encode)]
        )
        context = dy.concatenate([att_context, global_max, global_min])
        return context

def test_item(model, document, review, review_option, reviw_contribution):
    if review_option == "OnlyReview":
        context = get_context(model, review.preprocessed_sentences)
    elif review_option == "OnlyDocument":
        context = get_context(model, document.preprocessed_sentences)
    else: #Both Review And Document
        c1 = get_context(model, document.preprocessed_sentences)
        c2 = get_context(model, review.preprocessed_sentences)
        context = (c1* (1-reviw_contribution)) + (c2 * reviw_contribution)
    y_pred = dy.logistic((model.mlp_w * context) + model.mlp_b)
    document.prediction_result = y_pred.scalar_value()
    dy.renew_cg()
    return document.prediction_result


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


def test_all(args, model, data, review_option, reviw_contribution=0):
    def pr_roc_auc(predictions, gold):
        y_true = np.array(gold)
        y_scores = np.array(predictions)
        roc_auc = roc_auc_score(y_true, y_scores)
        pr_auc = average_precision_score(y_true, y_scores)
        return roc_auc, pr_auc

    write_file(args.outdir, "Predicting..")

    predictions, gold = [], []
    for index, document in enumerate(data.test_entries):
        if document.app_id in data.reviews:
            pred = test_item(model, document, data.reviews[document.app_id], review_option, reviw_contribution)
        else:
            pred = test_item(model, document, document, review_option, reviw_contribution)
        predictions.append(pred)
        gold.append(document.permissions[args.permission_type])
    return pr_roc_auc(predictions, gold)


def kfold_validation(args, data, review_option):
    data.entries = np.array(data.entries)
    random.shuffle(data.entries)
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    train_test_split = list(kfold.split(data.entries))
    review_contributions = []
    roc_values = []
    pr_values = []
    for reviw_contribution in np.arange(0, 1, 0.05):
        roc_l, pr_l = [], []
        for foldid, (train, test) in enumerate(train_test_split):
            write_file(args.outdir, "Fold {}".format(foldid + 1))

            model = Model(data, args)
            data.train_entries = data.entries[train]
            data.test_entries = data.entries[test]
            max_roc_auc, max_pr_auc = 0, 0
            for epoch in range(args.num_epoch):
                base = os.path.basename(args.model_checkpoint)
                directory = os.path.dirname(args.model_checkpoint)
                path = os.path.join(directory, "{}.{}-{}".format(foldid, epoch, base))    
                if os.path.exists(path):
                    model.load(path)
                else:
                    train_all(args, model, data)
                    model.save(path)

                roc_auc, pr_auc = test_all(args, model, data, review_option)
                if pr_auc > max_pr_auc:
                    max_pr_auc = pr_auc
                    max_roc_auc = roc_auc
                write_file(
                    args.outdir, "Epoch {} ROC {}  PR {}".format(epoch + 1, roc_auc, pr_auc)
                )

            roc_l.append(max_roc_auc)
            pr_l.append(max_pr_auc)
        write_file(
            args.outdir, "Summary : ROC {} PR {}".format(np.mean(roc_l), np.mean(pr_l))
        )
        review_contributions.append(reviw_contribution)
        roc_values.append(np.mean(roc_l))
        pr_values.append(np.mean(pr_l))
    return review_contributions, roc_values, pr_values

def run(args):
    data = Data()
    data.load(args.saved_data)

    review_contributions, roc_values, pr_values = kfold_validation(args, data, args.review_option)
    import pdb
    pdb.set_trace()
    fig, ax = plt.subplots()
    ax.plot(review_contributions, roc_values, '-b', label="ROC-AUC")
    ax.plot(review_contributions, pr_values, '--r', label="PR-AUC")
    ax.set(xlabel='Review Contribution (s)', ylabel='Score (mV)',
       title='ROC-AUC and PR-AUC scores according to Review Contribution')
    leg = ax.legend(loc='upper right', shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    ax.grid()
    fig.savefig("{}.png".format(args.permission_type))

    
