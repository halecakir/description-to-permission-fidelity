"""TODO"""
import os
import string

import stanfordnlp
from nltk import sent_tokenize
from nltk.corpus import stopwords

DIR_NAME = os.path.dirname(__file__)
MODELS_DIR = os.path.join(DIR_NAME, "../../../data/models")
stanfordnlp.download('en', MODELS_DIR)
NLP = stanfordnlp.Pipeline(processors='tokenize,depparse',
                           models_dir=MODELS_DIR,
                           treebank='en_ewt', use_gpu=True, pos_batch_size=3000)


class NLPUtils:
    """TODO"""
    @staticmethod
    def sentence_tokenization(text):
        """TODO"""
        lines = text.split("\n")
        sentences = []
        for line in lines:
            for sentence in sent_tokenize(line):
                sentences.append(sentence)
        return sentences

    @staticmethod
    def remove_hyperlinks(text):
        """TODO"""
        import re
        regex = r"((https?:\/\/)?[^\s]+\.[^\s]+)"
        text = re.sub(regex, '', text)
        return text

    @staticmethod
    def stopword_elimination(sentence):
        """TODO"""
        return [word for word in sentence if word not in stopwords.words('english')]

    @staticmethod
    def nonalpha_removal(sentence):
        """TODO"""
        return [word for word in sentence if word.isalpha()]

    @staticmethod
    def punctuation_removal(token):
        """TODO"""
        translator = str.maketrans('', '', string.punctuation)
        return token.translate(translator)

    @staticmethod
    def to_lower(token, lower):
        """TODO"""
        return token.lower() if lower else token

    @staticmethod
    def word_tokenization(sentence):
        """TODO"""
        doc = NLP(sentence)
        return [token.text for token in doc.sentences[0].tokens]

    @staticmethod
    def dependency_parse(sentence):
        """TODO"""
        doc = NLP(sentence)
        return [dep for dep in doc.sentences[0].dependencies]
