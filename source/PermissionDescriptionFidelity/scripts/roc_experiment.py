import sys
from numpy import inf
from scripts.similarity_experiment import SimilarityExperiment

from model.rnn_model import RNNModel
from utils.io_utils import IOUtils

"""
class ArgumentParser():
    permission_type = "READ_CONTACTS"
    train = "/home/huseyinalecakir/Security/data/ac-net/ACNET_DATASET.csv"
    train_file_type = "acnet"
    test = "/home/huseyinalecakir/Security/data/whyper/Read_Contacts.csv"
    test_file_type = "whyper"
    external_embedding = "/home/huseyinalecakir/Security/data/{}".format("cc.en.300.bin")
    external_embedding_type = "fasttext"
    wembedding_dims = 300
    lstm_dims = 128
    sequence_type = "windowed"
    window_size = 2
    saved_parameters_dir = "/home/huseyinalecakir/Security/data/saved_parameters/{}/{}".format(permission_type, external_embedding_type)
    saved_prevectors = "embeddings.pickle"
    saved_vocab_test = "whyper-vocab.txt"
    saved_vocab_train = "acnet-vocab.txt"
    saved_preprocessed_whyper = "whyper-preprocessed.txt"
    saved_preprocessed_acnet = "acnet-preprocessed.txt"
    outdir = "./test/{}/{}".format(permission_type, external_embedding_type)
"""

import random
import os
import csv

import dynet_config
# Declare GPU as the default device type
dynet_config.set_gpu()
# Set some parameters manualy
dynet_config.set(mem=400, random_seed=123456789)
# Initialize dynet import using above configuration in the current scope

import scipy
import dynet as dy
import pandas as pd
import numpy as np

from utils.io_utils import IOUtils
from utils.nlp_utils import NLPUtils


random.seed(33)


class SentenceReport:
    """TODO"""
    def __init__(self, sentence, mark):
        self.mark = mark
        self.preprocessed_sentence = None
        self.sentence = sentence
        self.all_phrases = None
        self.feature_weights = None
        self.max_similarites = None
        self.prediction_result = None

        
class SimilarityExperiment:
    """TODO"""
    def __init__(self, w2i, options):
        print('Similarity Experiment - init')
        self.options = options
        self.model = dy.ParameterCollection()
        self.trainer = dy.AdamTrainer(self.model)
        self.w2i = w2i
        self.wdims = options.wembedding_dims
        self.ldims = options.lstm_dims

        self.ext_embeddings = None
        #Model Parameters
        self.wlookup = self.model.add_lookup_parameters((len(w2i), self.wdims))

        self.__load_model()

        self.phrase_rnn = [dy.VanillaLSTMBuilder(1, self.wdims, self.ldims, self.model)]
        self.mlp_w = self.model.add_parameters((1, self.ldims))
        self.mlp_b = self.model.add_parameters(1)

    def __load_model(self):
        if self.options.external_embedding is not None:
            if os.path.isfile(os.path.join(self.options.saved_parameters_dir,
                                           self.options.saved_prevectors)):
                self.__load_external_embeddings(os.path.join(self.options.saved_parameters_dir,
                                                             self.options.saved_prevectors),
                                                "pickle")
            else:
                self.__load_external_embeddings(self.options.external_embedding,
                                                self.options.external_embedding_type)
                self.__save_model()

    def __save_model(self):
        IOUtils.save_embeddings(os.path.join(self.options.saved_parameters_dir,
                                             self.options.saved_prevectors),
                                self.ext_embeddings)

    def __load_external_embeddings(self, embedding_file, embedding_file_type):
        ext_embeddings, ext_emb_dim = IOUtils.load_embeddings_file(
            embedding_file,
            embedding_file_type,
            lower=True)
        assert ext_emb_dim == self.wdims
        self.ext_embeddings = {}
        print("Initializing word embeddings by pre-trained vectors")
        count = 0
        for word in self.w2i:
            if word in ext_embeddings:
                count += 1
                self.ext_embeddings[word] = ext_embeddings[word]
                self.wlookup.init_row(self.w2i[word], ext_embeddings[word])
        print("Vocab size: %d; #words having pretrained vectors: %d" % (len(self.w2i), count))


def __split_into_windows(sentence, window_size):
    splitted_sentences = []
    if len(sentence) < window_size:
        splitted_sentences.append(sentence)
    else:
        for start in range(len(sentence) - window_size + 1):
            splitted_sentences.append([sentence[i+start] for i in range(window_size)])
    return splitted_sentences

def __find_all_possible_phrases(sentence, sentence_only=False):
    entries = sentence.split(" ")
    all_phrases = []
    if sentence_only:
        all_phrases.extend(__split_into_windows(entries, len(entries)))
    else:
        for windows_size in range(2, len(entries)+1):
            all_phrases.extend(__split_into_windows(entries, windows_size))
    return all_phrases

def __train(model, data):
    def encode_sequence(seq):
        rnn_forward = model.phrase_rnn[0].initial_state()
        for entry in seq:
            vec = model.wlookup[int(model.w2i.get(entry, 0))]
            rnn_forward = rnn_forward.add_input(vec)
        return rnn_forward.output()
    tagged_loss = 0
    untagged_loss = 0
    for index, sentence_report in enumerate(data):
        for phrase in sentence_report.all_phrases:
            loss = None
            encoded_phrase = encode_sequence(phrase)
            y_pred = dy.logistic((model.mlp_w*encoded_phrase) + model.mlp_b)

            if sentence_report.mark:
                loss = dy.binary_log_loss(y_pred, dy.scalarInput(1))
            else:
                loss = dy.binary_log_loss(y_pred, dy.scalarInput(0))

            if sentence_report.mark:
                tagged_loss += loss.scalar_value()/(index+1)
            else:
                untagged_loss += loss.scalar_value()/(index+1)
            loss.backward()
            model.trainer.update()
            dy.renew_cg()

def __predict(model, data):
    def encode_sequence(seq):
        rnn_forward = model.phrase_rnn[0].initial_state()
        for entry in seq:
            vec = model.wlookup[int(model.w2i.get(entry, 0))]
            rnn_forward = rnn_forward.add_input(vec)
        return rnn_forward.output()

    for _, sentence_report in enumerate(data):
        for phrase in sentence_report.all_phrases:
            encoded_phrase = encode_sequence(phrase)
            y_pred = dy.logistic((model.mlp_w*encoded_phrase) + model.mlp_b)
            sentence_report.prediction_result = y_pred.scalar_value()
            dy.renew_cg()

def __report_confusion_matrix(sentence_reports, threshold):
    results = {"TP":0, "TN":0, "FP":0, "FN":0, "Threshold":threshold}
    total = 0
    for report in sentence_reports:
        total += 1
        if report.mark:
            if report.prediction_result >= threshold:
                results["TP"] += 1
            else:
                results["FN"] += 1
        else:
            if report.prediction_result >= threshold:
                results["FP"] += 1
            else:
                results["TN"] += 1
    try:
        results["precision"] = results["TP"]/(results["TP"]+results["FP"])
        results["recall"] = results["TP"]/(results["TP"]+results["FN"])
        results["f1_score"] = 2*((results["precision"]*results["recall"])/(results["precision"]+results["recall"]))
        results["accuracy"] = (results["TP"]+results["TN"])/(results["TP"]+results["TN"]+results["FP"]+results["FN"])
    except ZeroDivisionError:
        results["precision"] = 0
        results["recall"] = 0
        results["f1_score"] = 0
        results["accuracy"] = 0
    return results

def __save_preprocessed_file(outfile, reports, permission):
    print("Saving {} for permission {}".format(outfile, permission))
    with open(outfile, "w") as target:
        for report in reports:
            if report.mark:
                target.write("{}||{}||{}\n".format(report.sentence, report.preprocessed_sentence, permission))
            else:
                target.write("{}||{}||{}\n".format(report.sentence, report.preprocessed_sentence, "NONE"))

def __load_preprocessed_file(infile, permission):
    print("Loading {} for permission {}".format(infile, permission))
    sentence_reports = []
    with open(infile, "r") as target:
        for line in target:
            t = line.strip().split("||")    
            sentence = t[0]
            mark = None
            if t[2] == permission:
                mark = True
            else:
                mark = False
            sentence_report = SentenceReport(sentence, mark)
            sentence_report.preprocessed_sentence = t[1]
            sentence_report.all_phrases = __find_all_possible_phrases( sentence_report.preprocessed_sentence,
                                                                       sentence_only=True)
            sentence_reports.append(sentence_report)
    return sentence_reports

def __load_row_whyper_file(infile):
    print("Loading row {}".format(infile))
    tagged_test_file = pd.read_csv(infile)
    test_sentence_reports = []
    
    #read and preprocess whyper sentences
    print("Reading Test Sentences")
    for idx, row in tagged_test_file.iterrows():
        sentence = str(row["Sentences"])
        if not sentence.startswith("#"):
            mark = None
            if "Manually Marked" in row:
                if row["Manually Marked"] == 1:
                    mark = True
                else:
                    mark = False
            else:
                raise Exception("Manually Marked label does not exist")
            sentence_report = SentenceReport(sentence, mark)
            sentence_report.preprocessed_sentence = " ".join(NLPUtils.preprocess_sentence(sentence_report.sentence))
            sentence_report.all_phrases = __find_all_possible_phrases(sentence_report.preprocessed_sentence,
                                                                      sentence_only=True)
            test_sentence_reports.append(sentence_report)
    return test_sentence_reports

def __load_row_acnet_file(infile, gold_permission):
    print("Loading row {} ".format(infile))
    #read training data
    print("Reading Train Sentences")
    tagged_train_file = pd.read_csv(infile)
    train_sententence_reports = []
    acnet_map = {"RECORD_AUDIO" : "MICROPHONE", "READ_CONTACTS": "CONTACTS", "READ_CALENDAR": "CALENDAR"}
    for idx, row in tagged_train_file.iterrows():
        sentence = row["sentence"]
        mark = None
        if row[acnet_map[gold_permission]] is 1:
            mark = True
        else:
            mark = False
        sentence_report = SentenceReport(sentence, mark)
        sentence_report.preprocessed_sentence = " ".join(NLPUtils.preprocess_sentence(sentence_report.sentence))
        sentence_report.all_phrases = __find_all_possible_phrases( sentence_report.preprocessed_sentence,
                                                                    sentence_only=True)
        train_sententence_reports.append(sentence_report)
    return train_sententence_reports

def __load_whyper_sentences(options, permission):
    if options.saved_preprocessed_whyper is not None:
        if os.path.isfile(os.path.join(options.saved_parameters_dir,
                                       options.saved_preprocessed_whyper)):
            return __load_preprocessed_file(os.path.join(options.saved_parameters_dir,
                                                         options.saved_preprocessed_whyper), permission)
        else:
            reports = __load_row_whyper_file(options.test)
            __save_preprocessed_file(os.path.join(options.saved_parameters_dir,
                                                  options.saved_preprocessed_whyper), reports, permission)
            return reports
    else:
        raise Exception("Set saved_preprocessed_whyper option")

def __load_acnet_sentences(options, permission):
    if options.saved_preprocessed_acnet is not None:
        if os.path.isfile(os.path.join(options.saved_parameters_dir,
                                       options.saved_preprocessed_acnet)):
            return __load_preprocessed_file(os.path.join(options.saved_parameters_dir,
                                                 options.saved_preprocessed_acnet), permission)
        else:
            reports = __load_row_acnet_file(options.train, permission)
            __save_preprocessed_file(os.path.join(options.saved_parameters_dir,
                                                  options.saved_preprocessed_acnet), reports, permission)
            return reports
    else:
        raise Exception("Set saved_preprocessed_acnet option")
def __load_sentences(options, permission, data_type):
    if data_type == "ACNET":
        return __load_acnet_sentences(options, permission)
    elif data_type == "WHYPER":
        return __load_whyper_sentences(options, permission)
    else:
        raise Exception("Unkown data type")


# In[4]:


def run(args):
    print('Extracting training vocabulary')
    train_w2i, _ = IOUtils.load_vocab(  args.train,
                                        args.train_file_type,
                                        args.saved_parameters_dir,
                                        args.saved_vocab_train,
                                        args.external_embedding,
                                        args.external_embedding_type,
                                        True)

    print('Extracting test vocabulary')
    test_w2i, _ = IOUtils.load_vocab(args.test,
                                     args.test_file_type,
                                     args.saved_parameters_dir,
                                     args.saved_vocab_test,
                                     args.external_embedding,
                                     args.external_embedding_type,
                                     True)

    #combine test&train vocabulary
    w2i = train_w2i
    for token in test_w2i:
        if token not in w2i:
            w2i[token] = len(w2i)


    import matplotlib.pyplot as plt
    from itertools import cycle

    from sklearn import svm, datasets
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import label_binarize
    from sklearn.multiclass import OneVsRestClassifier
    from scipy import interp
    from matplotlib import pyplot

    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
    from sklearn.model_selection import KFold

    roc_scores = []
    pr_scores = []
    
    whole_sentences = __load_sentences(args,  args.permission_type, "ACNET")
    whole_sentences = np.array(whole_sentences)
    random.shuffle(whole_sentences)

    kfold = KFold(10, True, 1)
    for train, test in kfold.split(whole_sentences):
        print('Similarity Experiment')

        model = SimilarityExperiment(w2i, args)
        test_sentences = whole_sentences[test]
        train_sentences = whole_sentences[train]

        __train(model, train_sentences)
        __predict(model, test_sentences)

        predictions = [r.prediction_result for r in test_sentences]
        gold = []
        for r in test_sentences:
            if r.mark:
                gold.append(1)
            else:
                gold.append(0)

        y_true = np.array(gold)
        y_scores = np.array(predictions)

        roc_auc = roc_auc_score(y_true, y_scores)
        pr_auc = average_precision_score(y_true, y_scores)  

        roc_scores.append(roc_auc)
        pr_scores.append(pr_auc)
        print(roc_auc, pr_auc)

    roc_pr_out_dir = os.path.join(model.options.outdir, "roc_auc.txt")
    with open(roc_pr_out_dir, "w") as target:
        target.write("ROC-AUC {}\n".format(sum(roc_scores)/len(roc_scores)))
        target.write("PR-AUC {}\n".format(sum(pr_scores)/len(pr_scores)))

