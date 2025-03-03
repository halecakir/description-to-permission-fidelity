import sys
import os
import csv
import random

import pickle
import scipy
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import optim

from utils.io_utils import IOUtils
from utils.nlp_utils import NLPUtils

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold

seed = 10

random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
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
            for entry in self.entries:
                entry.index_tensor = entry.index_tensor.to(device=device)
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


class Encoder(nn.Module):
    def __init__(self, opt, w2i):
        super(Encoder, self).__init__()
        self.opt = opt
        self.w2i = w2i

        self.gru = nn.GRU(
            self.opt.hidden_size, self.opt.hidden_size, batch_first=True
        )
        if opt.dropout > 0:
            self.dropout = nn.Dropout(opt.dropout)
        self.embedding = nn.Embedding(len(self.w2i), self.opt.hidden_size)
        self.__initParameters()

    def __initParameters(self):
        for name, param in self.named_parameters():
            if "bias" not in name and param.requires_grad:
                init.xavier_uniform_(param)

    def initalizedPretrainedEmbeddings(self, embeddings):
        weights_matrix = np.zeros(((len(self.w2i), self.opt.hidden_size)))
        for word in self.w2i:
            weights_matrix[self.w2i[word]] = embeddings[word]
        self.embedding.weight = nn.Parameter(torch.FloatTensor(weights_matrix))

    def forward(self, input_src):
        src_emb = self.embedding(input_src)  # batch_size x src_length x emb_size
        if self.opt.dropout > 0:
            src_emb = self.dropout(src_emb)
        outputs, hidden = self.gru(src_emb)
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.opt = opt
        self.hidden_size = opt.hidden_size
        self.linear = nn.Linear(self.hidden_size, opt.hidden_size)
        self.context = torch.rand(self.hidden_size, device=self.opt.device, requires_grad=True)
        self.__initParameters()

    def __initParameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                init.uniform_(param, -self.opt.init_weight, self.opt.init_weight)

    def forward(self, inputs):
        inputs = inputs.view(inputs.shape[1], -1)
        hidden_repr = self.linear(inputs)
        hidden_with_context = torch.exp(torch.mv(hidden_repr, self.context))
        normalized_weight = hidden_with_context/torch.sum(hidden_with_context)
        normalized_hidden_repr = torch.sum(inputs.t()*normalized_weight, dim=1)
        return normalized_hidden_repr

class Classifier(nn.Module):
    def __init__(self, opt):
        super(Classifier, self).__init__()
        self.opt = opt
        self.hidden_size = opt.hidden_size
        self.linear = nn.Linear(self.hidden_size, opt.output_size)

        if opt.dropout > 0:
            self.dropout = nn.Dropout(opt.dropout)

        self.sigmoid = nn.Sigmoid()
        self.__initParameters()

    def __initParameters(self):
        for name, param in self.named_parameters():
            if "bias" not in name and param.requires_grad:
                init.xavier_uniform_(param)

    def forward(self, prev_h):
        if self.opt.dropout > 0:
            prev_h = self.dropout(prev_h)
        h2y = self.linear(prev_h)
        pred = self.sigmoid(h2y)
        return pred

class Model:
    def __init__(self):
        self.opt = None
        self.encoders = {}
        self.attentions = {}
        self.classifier = None
        self.optimizer = None
        self.criterion = None

    def create(self, opt, data):
        self.opt = opt
        self.encoders["sentence"] = Encoder(self.opt, data.w2i).to(
            device=self.opt.device
        )
        self.attentions["word_attention"] = Attention(self.opt).to(
            device=self.opt.device
        )
        params = []
        for encoder in self.encoders:
            params += list(self.encoders[encoder].parameters())
        self.classifier = Classifier(self.opt).to(device=self.opt.device)
        params += list(self.classifier.parameters())
        self.optimizer = optim.Adam(params)
        self.criterion = nn.BCELoss()

    def train(self):
        for encoder in self.encoders:
            self.encoders[encoder].train()
        for attention in self.attentions:
            self.attentions[attention].train()
        self.classifier.train()

    def eval(self):
        for encoder in self.encoders:
            self.encoders[encoder].eval()
        for attention in self.attentions:
            self.attentions[attention].eval()
        self.classifier.eval()

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def grad_clip(self):
        for encoder in self.encoders:
            torch.nn.utils.clip_grad_value_(
                self.encoders[encoder].parameters(), self.opt.grad_clip
            )
        for attention in self.attentions:
            torch.nn.utils.clip_grad_value_(
                self.attentions[attention].parameters(), self.opt.grad_clip
            )
        torch.nn.utils.clip_grad_value_(
            self.classifier.parameters(), self.opt.grad_clip
        )

    def save(self, filename):
        checkpoint = {}
        checkpoint["opt"] = self.opt
        for encoder in self.encoders:
            checkpoint[encoder] = self.encoders[encoder].state_dict()
        for attention in self.attentions:
            checkpoint[attention] = self.attentions[attention].state_dict()
        checkpoint["classifier"] = self.classifier.state_dict()
        checkpoint["optimizer"] = self.optimizer.state_dict()
        torch.save(checkpoint, filename)

    def load(self, filename, data):
        checkpoint = torch.load(filename)
        opt = checkpoint["opt"]
        self.create(opt, data)
        for encoder in self.encoders:
            self.encoders[encoder].load_state_dict(checkpoint[encoder])
        for attention in self.attentions:
            self.attentions[attention].load_state_dict(checkpoint[attention])
        self.classifier.load_state_dict(checkpoint["classifier"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])


def write_file(filename, string):
    with open(filename, "a") as target:
        target.write("{}\n".format(string))
        target.flush()


def train_item(args, model, sentence):
    model.zero_grad()
    outputs, hidden = model.encoders["sentence"](sentence.index_tensor)
    context = model.attentions["word_attention"](outputs)
    pred = model.classifier(context)

    loss = model.criterion(
        pred,
        torch.tensor(
            [sentence.permissions[args.permission_type]], dtype=torch.float
        ).to(device=args.device),
    )
    loss.backward()

    if model.opt.grad_clip != -1:
        model.grad_clip()
    model.step()
    return loss


def test_item(model, sentence):
    outputs, hidden = model.encoders["sentence"](sentence.index_tensor)
    context = model.attentions["word_attention"](outputs)
    pred = model.classifier(context)
    return pred



def train_all(args, model, data):
    write_file(args.outdir, "Training...")

    model.train()
    losses = []
    for index, sentence in enumerate(data.train_entries):
        loss = train_item(args, model, sentence)
        if index != 0:
            if index % model.opt.print_every == 0:
                write_file(
                    args.outdir,
                    "Index {} Loss {}".format(
                        index, np.mean(losses[index - model.opt.print_every :])
                    ),
                )
        losses.append(loss.item())


def test_all(args, model, data):
    def pr_roc_auc(predictions, gold):
        y_true = np.array(gold)
        y_scores = np.array(predictions)
        roc_auc = roc_auc_score(y_true, y_scores)
        pr_auc = average_precision_score(y_true, y_scores)
        return roc_auc, pr_auc

    write_file(args.outdir, "Predicting..")

    predictions, gold = [], []
    model.eval()
    with torch.no_grad():
        for index, sentence in enumerate(data.test_entries):
            pred = test_item(model, sentence)
            predictions.append(pred.cpu())
            gold.append(sentence.permissions[args.permission_type])
    return pr_roc_auc(predictions, gold)


def kfold_validation(args, data):
    data.entries = np.array(data.entries)
    random.shuffle(data.entries)

    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    roc_l, pr_l = [], []
    for foldid, (train, test) in enumerate(kfold.split(data.entries)):
        write_file(args.outdir, "Fold {}".format(foldid + 1))

        model = Model()
        model.create(args, data)
        data.train_entries = data.entries[train]
        data.test_entries = data.entries[test]

        train_all(args, model, data)
        roc_auc, pr_auc = test_all(args, model, data)

        write_file(args.outdir, "ROC {} PR {}".format(roc_auc, pr_auc))
        roc_l.append(roc_auc)
        pr_l.append(pr_auc)
    write_file(
        args.outdir, "Summary : ROC {} PR {}".format(np.mean(roc_l), np.mean(pr_l))
    )


def run(args):
    data = Data()
    data.load(args.saved_data)
    data.to(args.device)

    kfold_validation(args, data)
