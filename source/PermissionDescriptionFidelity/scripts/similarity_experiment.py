"""TODO"""
import random
import os

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

random.seed(33)


class SentenceReport:
    """TODO"""
    def __init__(self, sentence, mark):
        self.mark = mark
        self.sentence = sentence
        self.all_phrases = []
        self.max_similarites = {"RNN" : {"READ_CALENDAR" : {"similarity" : 0, "phrase" : ""},
                                         "READ_CONTACTS" : {"similarity" : 0, "phrase" : ""},
                                         "RECORD_AUDIO" :  {"similarity" : 0, "phrase" : ""}},
                                "ADDITION" : {"READ_CALENDAR" : {"similarity" : 0, "phrase" : ""},
                                              "READ_CONTACTS" : {"similarity" : 0, "phrase" : ""},
                                              "RECORD_AUDIO"  :  {"similarity" : 0, "phrase" : ""}}}


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
        if encode_type == "RNN":
            dy.renew_cg()
            rnn_forward = self.phrase_rnn[0].initial_state()
            for entry in phrase:
                vec = self.wlookup[int(self.w2i.get(entry, 0))]
                rnn_forward = rnn_forward.add_input(vec)
            return rnn_forward.output().npvalue()
        elif encode_type == "ADDITION":
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
        normalized_similarity = (dot(vec1, vec2)/(norm(vec1)*norm(vec2)) + 1)/2
        return normalized_similarity

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

    def __find_all_possible_phrases(self, sentence, mark):
        sentence_report = SentenceReport(sentence, mark)
        entries = self.__split_into_entries(sentence)
        for windows_size in range(2, len(entries)+1):
            sentence_report.all_phrases.extend(self.__split_into_windows(entries, windows_size))
        return sentence_report

    def __find_max_similarities(self, sentence_report):
        def encode_permissions(encode_type):
            permissions = {}
            permissions["READ_CALENDAR"] = self.__encode_phrase(["read", "calendar"], encode_type)
            permissions["READ_CONTACTS"] = self.__encode_phrase(["read", "contacts"], encode_type)
            permissions["RECORD_AUDIO"] = self.__encode_phrase(["record", "audio"], encode_type)
            return permissions

        for encode_type in sentence_report.max_similarites:
            encoded_permissions = encode_permissions(encode_type)
            for part in sentence_report.all_phrases:
                encoded_phrase = self.__encode_phrase(part, encode_type)
                for perm in encoded_permissions:
                    similarity_result = self._cos_similarity(encoded_phrase, encoded_permissions[perm])
                    if sentence_report.max_similarites[encode_type][perm]["similarity"] < similarity_result:
                        sentence_report.max_similarites[encode_type][perm]["similarity"] = similarity_result
                        sentence_report.max_similarites[encode_type][perm]["phrase"] = part
        return sentence_report

    def __dump_detailed_analysis(self, reports, file_name, reported_permission):
        with open(file_name, "w") as target:
            for report in reports:
                tag = "POSITIVE" if report.mark else "NEGATIVE"
                target.write("{} Sentence '{}' - Hantagged Permission {}\n".format(tag, report.sentence, reported_permission))
                for composition_type in report.max_similarites:
                    target.write("\t{} composition resulreported_permissionts : \n".format(composition_type))
                    for permission in report.max_similarites[composition_type]:
                        simimarity =    \
                            report.max_similarites[composition_type][permission]["similarity"]
                        phrase =    \
                            report.max_similarites[composition_type][permission]["phrase"]
                        target.write("\t\t{0} : {1:.3f}\t{2}\n".format(permission, simimarity, phrase))
                target.write("\n")

    def __linearized_similarity_values(self, reports):
        values = {"POSITIVE": {}, "NEGATIVE": {}}
        for report in reports:
            report_tag = "POSITIVE" if report.mark else "NEGATIVE"
            for composition_type in report.max_similarites:
                if composition_type not in values[report_tag]:
                    values[report_tag][composition_type] = {}
                for permission in report.max_similarites[composition_type]:
                    if permission not in values[report_tag][composition_type]:
                        values[report_tag][composition_type][permission] = []
                    similarity = report.max_similarites[composition_type][permission]["similarity"]
                    values[report_tag][composition_type][permission].append(similarity)
        return values

    def __compute_all_desriptive_statistics(self, values):
        def compute_descriptive_statistics(array):
            stats = {}
            descriptive_stats = scipy.stats.describe(array)
            stats["count"] = len(array)
            stats["mean"] = descriptive_stats.mean
            stats["minmax"] = descriptive_stats.minmax
            stats["std"] = np.std(array)
            return stats

        stats = {}
        for tag in values:
            if tag not in stats:
                stats[tag] = {}
            for composition_type in values[tag]:
                if composition_type not in stats[tag]:
                    stats[tag][composition_type] = {}
                for permission in values[tag][composition_type]:
                    if permission not in stats[tag][composition_type]:
                        stats[tag][composition_type][permission] = {}
                    linearized_values = values[tag][composition_type][permission]
                    stats[tag][composition_type][permission] = compute_descriptive_statistics(linearized_values)
        return stats

    def __write_all_stats(self, stats, file_name):
        with open(file_name, "w") as target:
            for tag_idx, tag in enumerate(stats):
                target.write("{}. {} Examples\n".format(tag_idx+1, tag))
                for c_type_idx, composition_type in enumerate(stats[tag]):
                    target.write("\t{}.{} {} Compostion\n".format(tag_idx+1, c_type_idx+1, composition_type))
                    for perm_idx, permission in enumerate(stats[tag][composition_type]):
                        target.write("\t\t{}.{}.{} {} Permission\n".format(tag_idx+1, c_type_idx+1, perm_idx+1, permission))
                        for stat in stats[tag][composition_type][permission]:
                            val = stats[tag][composition_type][permission][stat]
                            target.write("\t\t\t{} : {}\n".format(stat, val))
                target.write("\n\n")

    def __draw_distribution(self, data, axis_label, file_name):
        from matplotlib import pyplot as plt
        import seaborn as sns
        sns.set_style('darkgrid')
        sns.distplot(data, axlabel=axis_label)
        plt.savefig(file_name)
        plt.clf()

    def __draw_charts(self, values, gold_permission):
        for tag in values:
            for composition_type in values[tag]:
                for permission in values[tag][composition_type]:
                    self.__draw_distribution(values[tag][composition_type][permission],
                                             "{}_{}_{}".format(tag,
                                                               composition_type,
                                                               permission),
                                             "{}_{}_{}_(gold-{}).png".format(tag,
                                                                            composition_type,
                                                                            permission,
                                                                            gold_permission))
    def __normalize_similarity_values(self, values):
        normalized_values = {}
        for tag in values:
            if tag not in normalized_values:
                normalized_values[tag] = {}
            for composition_type in values[tag]:
                if composition_type not in normalized_values[tag]:
                    normalized_values[tag][composition_type] = {}
                for permission in values[tag][composition_type]:
                    if permission not in normalized_values[tag][composition_type]:
                        normalized_values[tag][composition_type][permission] = []
                    for sim in values[tag][composition_type][permission]:
                        if sim != 0:
                         normalized_values[tag][composition_type][permission].append(sim)
        return normalized_values


    def __find_optimized_threshold(self, values, composition_type, gold_permission):
        with open("{}_{}_threshold_results.txt".format(gold_permission, composition_type), "w") as target:
            tp = values["POSITIVE"][composition_type][gold_permission.upper()]
            tn = values["NEGATIVE"][composition_type][gold_permission.upper()]
            fn = []
            for perm in values["POSITIVE"][composition_type]:
                if perm != gold_permission.upper():
                    fn.extend(values["POSITIVE"][composition_type][perm])
            target.write("TP {} - TN {} - FN {}\n".format(len(tp), len(tn), len(fn)))
            target.write("Applying Threshold:\n")
            best_threshold_tp_tn = 0
            best_threshold_tp_fn = 0
            best_threshold_tp_tn_fn = 0
            thresh_tp_tn = 0
            thresh_tp_fn = 0
            thresh_tp_tn_fn = 0

            for threshold in np.arange(0.01, 0.99, 0.01):
                thresholded_tp = len(list(filter(lambda x: x > threshold, tp)))
                thresholded_tn = len(list(filter(lambda x: x < threshold, tn)))
                thresholded_fn = len(list(filter(lambda x: x < threshold, fn)))
                target.write("Threshold : {}\n".format(threshold))
                target.write("\tCount : TP {} - TN {} - FN {}\n".format(thresholded_tp, thresholded_tn, thresholded_fn))
                target.write("\tRatio : TP {} - TN {} - FN {}\n".format(thresholded_tp/len(tp)*100,
                                                                      thresholded_tn/len(tn)*100,
                                                                      thresholded_fn/len(fn)*100))

                if thresholded_tp + thresholded_tn > best_threshold_tp_tn:
                    best_threshold_tp_tn = thresholded_tp + thresholded_tn
                    thresh_tp_tn = threshold
                    target.write("\t\t{} : {}\n".format("thresh_tp_tn", thresh_tp_tn))
                if thresholded_tp + thresholded_fn > best_threshold_tp_fn:
                    best_threshold_tp_fn = thresholded_tp + thresholded_fn
                    thresh_tp_fn = threshold
                    target.write("\t\t{} : {}\n".format("thresh_tp_fn", thresh_tp_fn))
                if thresholded_tp + thresholded_fn + thresholded_tn > best_threshold_tp_tn_fn:
                    best_threshold_tp_tn_fn = thresholded_tp + thresholded_fn + thresholded_tn
                    thresh_tp_tn_fn = threshold
                    target.write("\t\t{} : {}\n".format("thresh_tp_tn_fn", thresh_tp_tn_fn))

            target.write("\n--Best Thresholded Parameters--\n")
            target.write("best_threshold_tp_tn {} : {}\n".format(best_threshold_tp_tn, thresh_tp_tn))
            target.write("best_threshold_tp_fn {} : {}\n".format(best_threshold_tp_fn, thresh_tp_fn))
            target.write("best_threshold_tp_tn_fn {} : {}\n".format(best_threshold_tp_tn_fn, thresh_tp_tn_fn))


    def run(self):
        """TODO"""
        print('Similarity Experiment - run')
        excel_file = self.options.train
        data_frame = pd.read_excel(excel_file)
        tagged_read_calendar = data_frame[data_frame["Manually Marked"].isin([0, 1, 2, 3])]

        sentence_similarity_reports = []
        for index, (_, row) in zip(range(10), tagged_read_calendar.iterrows()):
            sentence = row["Sentences"]
            mark = False if row["Manually Marked"] is 0 else True
            sentence_report = self.__find_all_possible_phrases(sentence, mark)
            sentence_similarity_report = self.__find_max_similarities(sentence_report)
            sentence_similarity_reports.append(sentence_similarity_report)

        gold_permission = os.path.basename(excel_file).split('.')[0].lower()

        self.__dump_detailed_analysis(sentence_similarity_reports, "{}_analysis.txt".format(gold_permission), gold_permission.upper())

        values = self.__linearized_similarity_values(sentence_similarity_reports)
        stats = self.__compute_all_desriptive_statistics(values)
        self.__write_all_stats(stats, "{}_stats.txt".format(gold_permission))

        #self.__draw_charts(values, gold_permission)
        self.__find_optimized_threshold(values, "ADDITION", gold_permission)