import random
import numpy as np

import torch.nn as nn
import torch.nn.init as init
from torch import optim

from utils.io_utils import IOUtils
from utils.nlp_utils import NLPUtils
from utils.common import *

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold

seed = 10

random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


class TorchOptions:
    embedding_size = 300
    hidden_size = 300
    output_size = 1
    init_weight = 0.08
    decay_rate = 0.985
    learning_rate = 0.0001
    plot_every = 2500
    print_every = 50
    grad_clip = 5
    dropout = 0
    dropoutrec = 0
    learning_rate_decay = 1  # 0.985
    learning_rate_decay_after = 1


class Encoder(nn.Module):
    def __init__(self, opt, w2i, embeddings):
        super(Encoder, self).__init__()
        self.opt = opt
        self.w2i = w2i

        self.embedding = None
        self.lstm = nn.LSTM(
            self.opt.embedding_size, self.opt.hidden_size, batch_first=True
        )
        if opt.dropout > 0:
            self.dropout = nn.Dropout(opt.dropout)
        self.__initParameters()
        self.embedding = nn.Embedding.from_pretrained(
            self.__initalizedPretrainedEmbeddings(embeddings)
        )

    def __initParameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                init.xavier_normal_(param)

    def __initalizedPretrainedEmbeddings(self, embeddings):
        weights_matrix = np.zeros(((len(self.w2i), self.opt.hidden_size)))
        for word in self.w2i:
            weights_matrix[self.w2i[word]] = embeddings[word]
        return weights_matrix

    def forward(self, input_src):
        src_emb = self.embedding(input_src)  # batch_size x src_length x emb_size
        if self.opt.dropout > 0:
            src_emb = self.dropout(src_emb)
        outputs, (h, c) = self.lstm(src_emb)
        return outputs, (h, c)


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
            if param.requires_grad:
                init.uniform_(param, -self.opt.init_weight, self.opt.init_weight)

    def forward(self, prev_h):
        if self.opt.dropout > 0:
            prev_h = self.dropout(prev_h)
        h2y = self.linear(prev_h)
        pred = self.sigmoid(h2y)
        return pred


def train_item(opt, args, sentence, encoder, classifier, optimizer, criterion):
    optimizer.zero_grad()
    outputs, (hidden, cell) = encoder(sentence.index_tensor)

    pred = classifier(hidden)

    loss = criterion(
        pred,
        torch.tensor(
            [[[sentence.permissions[args.permission_type]]]], dtype=torch.float
        ),
    )
    loss.backward()

    if opt.grad_clip != -1:
        torch.nn.utils.clip_grad_value_(encoder.parameters(), opt.grad_clip)
        torch.nn.utils.clip_grad_value_(classifier.parameters(), opt.grad_clip)
    optimizer.step()
    return loss


def predict(opt, sentence, encoder, classifier):
    outputs, (hidden, cell) = encoder(sentence.index_tensor)
    pred = classifier(hidden)
    return pred


def pr_roc_auc(predictions, gold):
    y_true = np.array(gold)
    y_scores = np.array(predictions)
    roc_auc = roc_auc_score(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    return roc_auc, pr_auc


def train_and_test(opt, args, w2i, train_data, test_data, ext_embeddings):
    encoder = Encoder(opt, w2i, ext_embeddings)

    classifier = Classifier(opt)

    params = list(encoder.parameters()) + list(classifier.parameters())
    optimizer = optim.Adam(params)
    criterion = nn.BCELoss()

    losses = []
    print("Training...")
    encoder.train()
    classifier.train()
    for index, sentence in enumerate(train_data):
        loss = train_item(
            opt, args, sentence, encoder, classifier, optimizer, criterion
        )
        if index != 0:
            if index % opt.print_every == 0:
                print(
                    "Index {} Loss {}".format(
                        index, np.mean(losses[index - opt.print_every :])
                    )
                )
        losses.append(loss.item())

    print("Predicting..")
    encoder.eval()
    classifier.eval()
    predictions = []
    gold = []
    with torch.no_grad():
        for index, sentence in enumerate(test_data):
            pred = predict(opt, sentence, encoder, classifier)
            predictions.append(pred)
            gold.append(sentence.permissions[args.permission_type])

    return pr_roc_auc(predictions, gold)


def kfold_validation(args, opt, ext_embeddings, sentences, w2i):
    documents = np.array(sentences)
    random.shuffle(documents)

    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    roc_l, pr_l = [], []
    for foldid, (train, test) in enumerate(kfold.split(documents)):
        print("Fold {}".format(foldid))
        train_data = documents[train]
        test_data = documents[test]
        roc, pr = train_and_test(opt, args, w2i, train_data, test_data, ext_embeddings)
        print("ROC {} PR {}".format(roc, pr))
        roc_l.append(roc)
        pr_l.append(pr)

    print("Summary : ROC {} PR {}".format(np.mean(roc_l), np.mean(pr_l)))


def run(args):
    opt = TorchOptions()

    ext_embeddings, sentences, w2i = load_data(args.saved_data)
    reviews = load_data(args.saved_review)

    kfold_validation(args, opt, ext_embeddings, sentences, w2i)
