"""TODO"""
import csv
import os
import pickle
from collections import Counter

import numpy as np
import xlrd
from gensim.models import KeyedVectors
from gensim.models.wrappers import FastText

from .nlp_utils import NLPUtils


class Application:
    """TODO"""
    def __init__(self, app_id, description, permissions):
        self.app_id = app_id
        self.description = description
        self.permissions = permissions


class Description:
    """TODO"""
    def __init__(self):
        self.sentences = []
        self.phrases = []


class WhyperDescription(Description):
    """TODO"""
    def __init__(self):
        super().__init__()
        self.manual_marked = []
        self.key_based = []
        self.whyper_tool = []


class UntaggedDescription(Description):
    """TODO"""
    pass


class Permission:
    """TODO"""
    def __init__(self, permission_type, permission_phrase):
        self.ptype = permission_type
        self.pphrase = permission_phrase

    def __repr__(self):
        return "Permission({}, {})".format(self.ptype, " ".join(self.pphrase))

    def __eq__(self, other):
        if isinstance(other, Permission):
            return self.ptype == other.ptype
        else:
            return False

    def __hash__(self):
        return hash(self.__repr__())


class IOUtils:
    """TODO"""
    @staticmethod
    def vocab(file_path, file_type, lower):
        """Return the set of distinct tokens from given dataset."""

        words_count = Counter()
        permissions = []

        # Use below subset of permissions
        handtagged_permissions = ["READ_CALENDAR",
                                  "READ_CONTACTS", "RECORD_AUDIO"]
        for permission in handtagged_permissions:
            for token in permission.split("_"):
                words_count.update([NLPUtils.to_lower(token, lower)])

        if file_type == "csv":
            with open(file_path) as csv_file:
                reader = csv.reader(csv_file)
                next(reader)  # skip header
                for row in reader:
                    text = row[1]
                    for sentence in text.split("%%"):
                        words_count.update([NLPUtils.to_lower(w, lower)
                                            for w in sentence.split(" ")])
        elif file_type == "excel":
            loc = (file_path)
            workbook = xlrd.open_workbook(loc)
            sheet = workbook.sheet_by_index(0)
            sharp_count = 0
            for i in range(sheet.nrows):
                sentence = sheet.cell_value(i, 0)
                if not sentence.startswith("#"):
                    if sharp_count != 0 and sharp_count % 2 == 0:
                        sentence = sentence.strip()
                        for token in NLPUtils.word_tokenization(sentence):
                            words_count.update([NLPUtils.to_lower(token, lower)])
        else:
            raise Exception("Unsupported file type.")

        return words_count.keys(), \
               {w: i for i, w in enumerate(list(words_count.keys()))}, \
               permissions

    @staticmethod
    def __read_file_csv(file_path, lower):
        app_id = 0
        with open(file_path) as csv_file:
            reader = csv.reader(csv_file)
            next(reader)  # skip header
            for row in reader:
                app_id = row[0]
                description = row[1]
                permissions = set()
                for permission in row[2].strip().split("%%"):
                    ptype = NLPUtils.to_lower(permission, lower)
                    pphrase = [NLPUtils.to_lower(t, lower)
                               for t in permission.split("_")]
                    perm = Permission(ptype, pphrase)
                    permissions.add(perm)

                untagged_decription = UntaggedDescription()
                for sentence in description.split("%%"):
                    untagged_decription.sentences.append(sentence.strip())
                yield Application(app_id, untagged_decription, permissions)

    @staticmethod
    def __read_file_excel(file_path, lower):
        app_id = ""
        permission_title = file_path.split("/")[-1].split(".")[0]

        sharp_count = 0
        permissions = set()
        whyper_decription = WhyperDescription()

        loc = (file_path)
        workbook = xlrd.open_workbook(loc)
        sheet = workbook.sheet_by_index(0)
        for i in range(sheet.nrows):
            sentence = sheet.cell_value(i, 0)
            if sentence.startswith("#"):
                sharp_count += 1
                if sharp_count % 2 == 1:
                    if sharp_count != 1:
                        yield Application(app_id, whyper_decription, permissions)

                    # Document init values
                    # line can start with ## or #
                    app_id = sentence.split("#")[-1]
                    permissions = set()
                    whyper_decription = WhyperDescription()

                    # Permissions for apk
                    ptype = NLPUtils.to_lower(permission_title, lower)
                    pphrase = [NLPUtils.to_lower(t, lower)
                               for t in permission_title.split("_")]
                    perm = Permission(ptype, pphrase)
                    permissions.add(perm)
            else:
                if sharp_count != 0 and sharp_count % 2 == 0:
                    whyper_decription.sentences.append(sentence.strip())
                    whyper_decription.manual_marked.append(
                        sheet.cell_value(i, 1))
                    whyper_decription.key_based.append(sheet.cell_value(i, 2))
                    whyper_decription.whyper_tool.append(
                        sheet.cell_value(i, 3))

        yield Application(app_id, whyper_decription, permissions)
        workbook.release_resources()
        del workbook

    @staticmethod
    def __read_file(file_path, file_type, lower):
        if file_type == "csv":
            return IOUtils.__read_file_csv(file_path, lower)
        elif file_type == "excel":
            return IOUtils.__read_file_excel(file_path, lower)
        else:
            raise Exception("Unsupported file type.")

    @staticmethod
    def get_data(file_path, sequence_type, file_type, window_size, lower):
        """TODO"""
        if sequence_type == "raw":
            return IOUtils.__read_file_raw(file_path, file_type, lower)
        elif sequence_type == "dependency":
            return IOUtils.__read_file_dependency(file_path, file_type, lower)
        elif sequence_type == "windowed":
            return IOUtils.__read_file_window(file_path, file_type, window_size, lower)
        else:
            raise Exception("Unknown sequence type")

    @staticmethod
    def __read_file_raw(file_path, file_type, lower):
        for app in IOUtils.__read_file(file_path, file_type, lower):
            app.description.phrases = \
                [[NLPUtils.to_lower(w, lower) for w in NLPUtils.word_tokenization(sentence)] \
                                                    for sentence in app.description.sentences]
            yield app

    @staticmethod
    def __read_file_window(file_path, file_type, window_size, lower):
        for app in IOUtils.__read_file(file_path, file_type, lower):
            raw_sentences = [[NLPUtils.to_lower(w, lower) for w in NLPUtils.word_tokenization(
                sentence)] for sentence in app.description.sentences]
            app.description.phrases = IOUtils.__split_into_windows(
                raw_sentences, window_size)
            yield app

    @staticmethod
    def __read_file_dependency(file_path, file_type, lower):
        for app in IOUtils.__read_file(file_path, file_type, lower):
            app.description.phrases = IOUtils.__split_into_dependencies(
                app.description.sentences)
            yield app

    @staticmethod
    def __split_into_dependencies(sentences):
        splitted_sentences = []
        for sentence in sentences:
            dependencies = [[rel[1].text, rel[2].text]
                            for rel in NLPUtils.dependency_parse(sentence) if rel[1] != 'root']
            splitted_sentences.append(dependencies)
        return splitted_sentences

    @staticmethod
    def __split_into_windows(sentences, window_size):
        splitted_sentences = []
        for sentence in sentences:
            splitted_sentences.append([])
            if len(sentence) < window_size:
                splitted_sentences[-1].append(sentence)
            else:
                for start in range(len(sentence) - window_size + 1):
                    splitted_sentences[-1].append([sentence[i+start]
                                                   for i in range(window_size)])
        return splitted_sentences

    @staticmethod
    def load_embeddings_file(file_name, embedding_type, lower):
        """TODO"""
        if not os.path.isfile(file_name):
            raise Exception("{} does not exist".format(file_name))

        if embedding_type == "word2vec":
            model = KeyedVectors.load_word2vec_format(
                file_name, binary=True, unicode_errors="ignore")
            words = model.index2entity
        elif embedding_type == "fasttext":
            model = FastText.load_fasttext_format(file_name)
            words = [w for w in model.wv.vocab]
        elif embedding_type == "pickle":
            with open(file_name, 'rb') as stream:
                model = pickle.load(stream)
                words = model.keys()
        else:
            raise Exception("Unknown Embedding Type")

        if lower:
            vectors = {word.lower(): model[word] for word in words}
        else:
            vectors = {word: model[word] for word in words}

        if "UNK" not in vectors:
            unk = np.mean([vectors[word] for word in vectors.keys()], axis=0)
            vectors["UNK"] = unk

        return vectors, len(vectors["UNK"])
