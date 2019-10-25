import pandas as pd

from utils.nlp_utils import NLPUtils


class DocumentReport:
    def __init__(self, app_id):
        self.app_id = app_id
        self.permissions = {}
        self.preprocessed_sentences = []
        self.sentences = []
        self.prediction_result = None
        self.index_tensors = None


class Review:
    def __init__(self, sentence, score):
        self.sentence = sentence
        self.preprocessed_sentence = None
        self.score = score
        self.index_tensor = None
        self.prediction_result = None


class SentenceReport:
    def __init__(self, id, sentence):
        self.app_id = id
        self.sentence = sentence
        self.permissions = {}
        self.preprocessed_sentence = None
        self.prediction_result = None
        self.index_tensor = None


def calculate_freqs(infile, stemmer, embeddings):
    tagged_train_file = pd.read_csv(infile)
    vocab_freq = {}
    for idx, row in tagged_train_file.iterrows():
        app_id = row["app_id"]
        sentence = row["sentence"]

        preprocessed = NLPUtils.preprocess_sentence(sentence, stemmer)
        for token in preprocessed:
            if token not in vocab_freq:
                vocab_freq[token] = 0
            vocab_freq[token] += 1
    return vocab_freq
