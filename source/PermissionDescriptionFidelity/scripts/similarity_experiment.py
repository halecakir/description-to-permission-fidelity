"""TODO"""
import random

import dynet_config
# Declare GPU as the default device type
dynet_config.set_gpu()
# Set some parameters manualy
dynet_config.set(mem=400, random_seed=123456789)
# Initialize dynet import using above configuration in the current scope

import dynet as dy
import pandas as pd

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
        self.encode_type = "addition"
        self.options = options
        self.model = dy.ParameterCollection()
        self.trainer = dy.AdamTrainer(self.model)
        self.w2i = w2i
        self.wdims = options.wembedding_dims
        self.ldims = options.lstm_dims
        #Model Parameters
        self.wlookup = self.model.add_lookup_parameters((len(w2i), self.wdims))

        if options.external_embedding is not None:
            self.__load_external_embeddings()

        self.phrase_rnn = [dy.SimpleRNNBuilder(1, self.wdims, self.ldims, self.model)]

    def __load_external_embeddings(self):
        ext_embeddings, ext_emb_dim = IOUtils.load_embeddings_file(
            self.options.external_embedding,
            self.options.external_embedding_type,
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

    def __encode_phrase(self, phrase, encode_type="rnn"):
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
                vec = self.wlookup[int(self.w2i.get(entry, 0))]
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

    def __encode_permissions(self):
        permissions = {}
        permissions["READ_CALENDAR"] = self.__encode_phrase(["read", "calendar"], encode_type=self.encode_type)
        permissions["READ_CONTACTS"] = self.__encode_phrase(["read", "contacts"], encode_type=self.encode_type)
        permissions["RECORD_AUDIO"] = self.__encode_phrase(["record", "audio"], encode_type=self.encode_type)
        return permissions

    def __find_all_parts_sim(self, sentence):
        permissions = self.__encode_permissions()
        all_sims = []
        sentence = self.__split_into_entries(sentence)
        splitted = []
        for windows_size in range(2, len(sentence)+1):
            splitted.extend(self.__split_into_windows(sentence, windows_size))
        for part in splitted:
            encoded = self.__encode_phrase(part, encode_type=self.encode_type)
            for perm in permissions:
                similarity_result = self._cos_similarity(encoded, permissions[perm])
                all_sims.append(Result(" ".join(part), perm, similarity_result))
        all_sims.sort(key=lambda x: x.similiarity, reverse=True)
        return all_sims

    def __report_sentece(self, sentence, sim_values, top=20):
        file = open("read_calendar_analysis_{}.txt".format(self.encode_type), "a")
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

    def run(self):
        """TODO"""
        excel_file = self.options.train
        data_frame = pd.read_excel(excel_file)
        tagged_read_calendar = data_frame[data_frame["Manually Marked"].isin([1, 2, 3])]

        for sentence in tagged_read_calendar["Sentences"]:
            sims = self.__find_all_parts_sim(sentence)
            self.__report_sentece(sentence, sims)
