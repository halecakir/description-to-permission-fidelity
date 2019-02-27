"""TODO"""
import random
import os

import dynet_config
# Declare GPU as the default device type
dynet_config.set_gpu()
# Set some parameters manualy
dynet_config.set(mem=400, random_seed=123456789)
# Initialize dynet import using above configuration in the current scope

import dynet as dy
import pandas as pd
from collections import Counter

from utils.io_utils import IOUtils

random.seed(33)


class Result:
    """TODO"""
    def __init__(self, phrase, perm, sim):
        self.phrase = phrase
        self.permission = perm
        self.similiarity = sim

class SimilarityExperiment:
    """TODO"""
    def __init__(self, w2i, options):
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

        self.phrase_rnn = [dy.SimpleRNNBuilder(1, self.wdims, self.ldims, self.model)]

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
        print("Initializing word embeddings by pre-trained vectors")
        count = 0
        for word in self.w2i:
            if word in ext_embeddings:
                count += 1
                self.wlookup.init_row(self.w2i[word], ext_embeddings[word])
        self.ext_embeddings = ext_embeddings
        print("Vocab size: %d; #words having pretrained vectors: %d" % (len(self.w2i), count))


    def __encode_phrase(self, phrase, encode_type):
        if encode_type == "rnn":
            dy.renew_cg()
            rnn_forward = self.phrase_rnn[0].initial_state()
            for entry in phrase:
                vec = self.wlookup[int(self.w2i.get(entry, 0))]
                rnn_forward = rnn_forward.add_input(vec)
            return rnn_forward.output().npvalue()
        elif encode_type == "addition":
            sum_vec = 0
            for entry in phrase:
                vec = self.wlookup[int(self.w2i.get(entry, 0))].npvalue()
                sum_vec += vec
            return sum_vec
        else:
            raise Exception("Undefined encode type")

    def _cos_similarity(self, vec1, vec2):
        from numpy import dot
        from numpy.linalg import norm
        return dot(vec1, vec2)/(norm(vec1)*norm(vec2))

    def __to_lower(self, phrase):
        return phrase.lower()

    def __split_into_entries(self, phrase):
        phrase = self.__to_lower(phrase)
        return phrase.strip().split(" ")

    def __split_into_windows(self, sentence, window_size):
        splitted_sentences = []
        if len(sentence) < window_size:
            splitted_sentences.append(sentence)
        else:
            for start in range(len(sentence) - window_size + 1):
                splitted_sentences.append([sentence[i+start] for i in range(window_size)])
        return splitted_sentences

    def __encode_permissions(self, encode_type):
        permissions = {}
        permissions["READ_CALENDAR"] = self.__encode_phrase(["read", "calendar"], encode_type)
        permissions["READ_CONTACTS"] = self.__encode_phrase(["read", "contacts"], encode_type)
        permissions["RECORD_AUDIO"] = self.__encode_phrase(["record", "audio"], encode_type)
        return permissions

    def __find_all_parts_sim(self, sentence, encode_type):
        permissions = self.__encode_permissions(encode_type)
        all_sims = []
        sentence = self.__split_into_entries(sentence)
        splitted = []
        for windows_size in range(2, len(sentence)+1):
            splitted.extend(self.__split_into_windows(sentence, windows_size))
        for part in splitted:
            encoded = self.__encode_phrase(part, encode_type)
            for perm in permissions:
                similarity_result = self._cos_similarity(encoded, permissions[perm])
                all_sims.append(Result(" ".join(part), perm, similarity_result))
        all_sims.sort(key=lambda x: x.similiarity, reverse=True)
        return all_sims

    def __report_sentence(self, sentence, sim_values, encode_type, top=20):
        file = open("read_calendar_analysis_{}.txt".format(encode_type), "a")
        file.write("Sentence '{}' - Hantagged Permission {}\n".format(sentence, "READ_CALENDAR"))
        for res, idx in zip(sim_values, range(top)):
            file.write("{}. {} vs '{}' = {}\n".format(idx+1,
                                                      res.permission,
                                                      res.phrase,
                                                      res.similiarity))
            #print("{}. {} vs '{}' = {}".format(idx+1, res.permission, res.phrase, res.similiarity))
        file.write("------\n\n\n")
        file.close()
        #print("------\n\n\n")


    def __draw_pie_chart(self, data, gold_permission):
        import matplotlib.ticker as ticker
        import matplotlib.cm as cm
        import matplotlib as mpl
        from matplotlib.gridspec import GridSpec

        import matplotlib.pyplot as plt

        import numpy as np

        rnn_counter = Counter(list(map(lambda r: r.permission, data["rnn"])))
        addition_counter = Counter(list(map(lambda r: r.permission, data["addition"])))

        rnn_counts = [rnn_counter[key] for key in rnn_counter.keys()]
        rnn_labels = rnn_counter.keys()
        rnn_explode = [0 if key != gold_permission else 0.1 for key in rnn_counter.keys()]

        addition_counts = [addition_counter[key] for key in addition_counter.keys()]
        addition_labels = addition_counter.keys()
        addition_explode = [0 if key != gold_permission else 0.1 for key in addition_counter.keys()]

        # Make square figures and axes
        plt.figure(1, figsize=(10, 6))
        the_grid = GridSpec(2, 2)

        cmap = plt.get_cmap('Spectral')
        all_labels = set(rnn_labels).union(set(addition_labels))
        colors = {label:cmap(i) for label, i in zip(all_labels, np.linspace(0, 1, 8))}

        plt.subplot(the_grid[0, 1], aspect=1, title='RNN Composition')
        plt.pie(rnn_counts, explode=rnn_explode, labels=rnn_labels, autopct='%1.1f%%', shadow=True, colors=[colors[l] for l in rnn_labels])

        plt.subplot(the_grid[0, 0], aspect=1, title='Vector Addition Composition')
        plt.pie(addition_counts, explode=addition_explode, labels=addition_labels, autopct='%.0f%%', shadow=True, colors=[colors[l] for l in addition_labels])

        plt.suptitle('Vector composition methods - Expected Permission : {}'.format(gold_permission), fontsize=16)
        plt.savefig("{}_pie.png".format(gold_permission))
        plt.close()

    def __draw_histogram(self, data, gold_permission):
        import numpy as np
        import matplotlib.pyplot as plt

        tp_reports_rnn = list(filter(lambda r: r.permission == gold_permission, data["rnn"]))
        tp_reports_addition = list(filter(lambda r: r.permission == gold_permission, data["addition"]))

        rnn_values = [r.similiarity for r in tp_reports_rnn]
        addition_values = [r.similiarity for r in tp_reports_addition]

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        ax0, ax1 = axes.flat

        ax0.hist(addition_values, bins='auto', normed=1, histtype='bar')
        ax0.set_title('Vector Addition Composition TP reports')


        ax1.hist(rnn_values, bins='auto', normed=1, histtype='bar')
        ax1.set_title('RNN Composition TP reports')

        fig.suptitle('Vector composition methods - Expected Permission : {}'.format(gold_permission), fontsize=16)
        fig.savefig("{}_histogram.png".format(gold_permission))
        plt.close()

    def __show_statistics(self, sim_reports, gold_permission):
        self.__draw_pie_chart(sim_reports, gold_permission)
        self.__draw_histogram(sim_reports, gold_permission)

    def run(self):
        """TODO"""
        excel_file = self.options.train
        data_frame = pd.read_excel(excel_file)
        tagged_read_calendar = data_frame[data_frame["Manually Marked"].isin([1, 2, 3])]

        method_reports = {"rnn" : [], "addition": []}
        for sentence in  tagged_read_calendar["Sentences"]:
            #RNN
            encode_type = "rnn"
            sims_rnn = self.__find_all_parts_sim(sentence, encode_type)
            self.__report_sentence(sentence, sims_rnn, encode_type)
            #Addition
            encode_type = "addition"
            sims_add = self.__find_all_parts_sim(sentence, encode_type)
            self.__report_sentence(sentence, sims_add, encode_type)
            if sims_rnn and sims_add:
                method_reports["rnn"].append(sims_rnn[0]) #first rank
                method_reports["addition"].append(sims_add[0]) #first rank
        self.__show_statistics(method_reports, "READ_CALENDAR")
