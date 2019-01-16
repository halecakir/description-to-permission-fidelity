import csv
import os
import pickle
import re
from collections import Counter

import numpy as np
import xlrd
from gensim.models import KeyedVectors
from gensim.models.wrappers import FastText
from nltk import sent_tokenize, word_tokenize
from stanfordcorenlp import StanfordCoreNLP

from decorators import logging


class Document:
    def __init__(self, doc_id, title, description, permissions, tags=None, raw_sentences=None):
        self.id = doc_id
        self.title = title
        self.permissions = permissions
        self.description = description 
        self.tags = tags
        self.raw_sentences = raw_sentences

    def __str__(self):
        print(self.title)
        print(self.permissions)



class Permission:
    def __init__(self, permission_type, permission_phrase):
        self.ptype = permission_type
        self.pphrase = permission_phrase

    def __repr__(self):
        return "Permission({}, {})".format(self.ptype, " ".join(self.pphrase))

    def __eq__(self, other):
        if isinstance(other, Permission):
            return ((self.ptype == other.ptype))
        else:
            return False

    def __hash__(self):
        return hash(self.__repr__())


class Utils:
    CORE_NLP_DIR = r'/home/huseyin/LIB/stanford-corenlp-full-2018-10-05'
    
    @staticmethod
    @logging
    def preprocess(text):
        paragrahps = text.split("\n")
        sentences = []
        for p in paragrahps:
            for s in sent_tokenize(p):
                sentences.append(s)
        return sentences

    @staticmethod
    @logging
    def remove_hyperlinks(text):
        regex = r"((https?:\/\/)?[^\s]+\.[^\s]+)"
        text = re.sub(regex, '', text)
        return text

    @staticmethod
    def to_lower(w, lower):
        return w.lower() if lower else w
    
    @staticmethod
    @logging
    def vocab(file_path, file_type="csv", lower=True):
        nlp = StanfordCoreNLP(Utils.CORE_NLP_DIR)
        wordsCount = Counter()
        permissions = []
        distincts_permissions = set()
        if file_type == "csv":
            with open(file_path) as f:
                reader = csv.reader(f)
                next(reader) # skip header
                for row in reader:
                    text = row[1]
                    for sentence in Utils.preprocess(text):
                        sentence = Utils.remove_hyperlinks(sentence)
                        for w in nlp.word_tokenize(sentence):
                            wordsCount.update([Utils.to_lower(w, lower)])
                        for p in  row[2].strip().split(","):
                            ptype = Utils.to_lower(p, lower)
                            if ptype not in distincts_permissions:
                                pphrase = [Utils.to_lower(t, lower) for t in p.split("_")]
                                perm = Permission(ptype, pphrase)
                                permissions.append(perm)
                                distincts_permissions.add(ptype)
                            for token in p.split("_"):
                                wordsCount.update([Utils.to_lower(token, lower)])
        elif file_type == "excel":
            handtagged_permissions = ["READ_CALENDAR", "READ_CONTACTS", "RECORD_AUDIO"]
            loc = (file_path)
            wb = xlrd.open_workbook(loc) 
            sheet = wb.sheet_by_index(0)
            sharp_count = 0
            apk_title = ""
            for i in range(sheet.nrows):
                sentence = sheet.cell_value(i, 0)
                if sentence.startswith("##"):
                    sharp_count += 1
                    if sharp_count % 2 == 1:
                        apk_title = sentence.split("##")[1] 
                else:
                    if sharp_count != 0 and sharp_count % 2 == 0:
                        sentence = sentence.strip()
                        for w in nlp.word_tokenize(sentence):
                            wordsCount.update([Utils.to_lower(w, lower)])
                        for p in handtagged_permissions:
                            ptype = Utils.to_lower(p, lower)
                            if ptype not in distincts_permissions:
                                pphrase = [Utils.to_lower(t, lower) for t in p.split("_")]
                                perm = Permission(ptype, pphrase)
                                permissions.append(perm)
                                distincts_permissions.add(ptype)
                                for token in p.split("_"):
                                    wordsCount.update([Utils.to_lower(token, lower)])            
        else:
            raise Exception("Unsupported file type.")
        nlp.close()
        return wordsCount.keys(), {w: i for i, w in enumerate(list(wordsCount.keys()))}, permissions

    @staticmethod
    @logging
    def read_file(file_path, w2i, file_type="csv", lower=True):
        data = []
        doc_id = 0
        if file_type == "csv":
            with open(file_path) as f:
                reader = csv.reader(f)
                next(reader) # skip header
                for row in reader:
                    doc_id += 1
                    title = row[0]
                    description = row[1]
                    permissions = set()
                    for p in  row[2].strip().split(","):
                        ptype = Utils.to_lower(p, lower)
                        pphrase = [Utils.to_lower(t, lower) for t in p.split("_")]
                        perm = Permission(ptype, pphrase)
                        permissions.add(perm)

                    sentences = []
                    raw_sentences = []
                    for sentence in Utils.preprocess(description):
                        sentence = Utils.remove_hyperlinks(sentence)
                        sentences.append(sentence.strip())
                        raw_sentences.append(sentence.strip())
                        ###sentences.append([Utils.to_lower(w, lower) for w in word_tokenize(sentence)])           
                    yield Document(doc_id, title, sentences, permissions,raw_sentences=raw_sentences)
                    
        elif file_type == "excel":
            permission_title = file_path.split("/")[-1].split(".")[0]
            loc = (file_path)
            wb = xlrd.open_workbook(loc) 
            sheet = wb.sheet_by_index(0)
            sharp_count = 0
            title = ""
            permissions = set()
            sentences = []
            raw_sentences = []
            tags = []
            for i in range(sheet.nrows):
                sentence = sheet.cell_value(i, 0)
                if sentence.startswith("##"):
                    sharp_count += 1
                    if sharp_count % 2 == 1:
                        if doc_id > 0:
                            yield Document(doc_id, title, sentences, permissions, tags, raw_sentences)
                        
                        #Document init values
                        title = sentence.split("##")[1]
                        permissions = set()
                        sentences = []
                        raw_sentences = []
                        tags = []
                        doc_id += 1
                        
                        # Permissions for apk
                        ptype = Utils.to_lower(permission_title, lower)
                        pphrase = [Utils.to_lower(t, lower) for t in permission_title.split("_")]
                        perm = Permission(ptype, pphrase)
                        permissions.add(perm)
                else:
                    if sharp_count != 0 and sharp_count % 2 == 0:
                        sentences.append(sentence.strip())
                        raw_sentences.append(sentence.strip())
                        ###sentences.append([Utils.to_lower(w, lower) for w in word_tokenize(sentence.strip())]) 
                        tags.append(int(sheet.cell_value(i, 1)))
                        
            yield Document(doc_id, title, sentences, permissions, tags, raw_sentences)
            wb.release_resources()
            del wb
        else:
            raise Exception("Unsupported file type.")
    
    @staticmethod
    @logging
    def get_data(file_path, w2i, sequence_type="dependency", file_type="csv", window_size=2, lower=True):
        nlp = StanfordCoreNLP(Utils.CORE_NLP_DIR)
        if sequence_type == "raw":
            return Utils.read_file_raw(file_path, w2i, nlp, file_type, lower)
        elif sequence_type == "dependency":
            return Utils.read_file_dependency(file_path, w2i, nlp, file_type, lower)
        elif sequence_type == "windowed":
            return Utils.read_file_window(file_path, w2i, nlp, file_type, window_size, lower)
        else:
            nlp.close()
            raise Exception("Unknown sequence type")
        
    @staticmethod
    def read_file_raw(file_path, w2i, nlp, file_type="csv", lower=True):
        for doc in Utils.read_file(file_path, w2i, file_type, lower):
            doc.description = [[Utils.to_lower(w, lower) for w in nlp.word_tokenize(sentence)] for sentence in doc.description]
            yield doc
        
    @staticmethod
    def read_file_window(file_path, w2i, nlp, file_type="csv", window_size=2, lower=True):
        for doc in Utils.read_file(file_path, w2i, file_type, lower):
            doc.description = [[Utils.to_lower(w, lower) for w in nlp.word_tokenize(sentence)] for sentence in doc.description]
            doc.description = Utils.split_into_windows(doc.description, window_size)
            yield doc
            
    @staticmethod
    def read_file_dependency(file_path, w2i, nlp, file_type="csv", lower=True):
        for doc in Utils.read_file(file_path, w2i, file_type, lower):
            doc.description = Utils.split_into_dependencies(doc.description, nlp)
            yield doc
            
    @staticmethod
    def split_into_dependencies(sentences, nlp):
        splitted_sentences = []
        for sentence in sentences:
            tokens = nlp.word_tokenize(sentence)
            s = [[tokens[rel[1]-1], tokens[rel[2]-1]] for rel in nlp.dependency_parse(sentence) if rel[0] != 'ROOT']
            splitted_sentences.append(s)
        return splitted_sentences
    
    @staticmethod
    def split_into_windows(sentences, window_size=2):
        splitted_sentences = []
        for sentence in sentences:
            splitted_sentences.append([])
            if len(sentence) < window_size:
                splitted_sentences[-1].append(sentence)
            else:
                for start in range(len(sentence) - window_size + 1):
                    splitted_sentences[-1].append([sentence[i+start] for i in range(window_size)])
        return splitted_sentences
    
    @staticmethod     
    def load_embeddings_file(file_name, embedding_type, lower=True):
        if not os.path.isfile(file_name):
            raise Exception("{} does not exist".format(file_name))
            
        if embedding_type == "word2vec":
            model = KeyedVectors.load_word2vec_format(file_name, binary=True, unicode_errors="ignore")
            words = model.index2entity
        elif embedding_type == "fasttext":
            model = FastText.load_fasttext_format(file_name)
            words = [w for w in model.wv.vocab]
        elif embedding_type == "pickle":
            with open(file_name,'rb') as fp:
                model = pickle.load(fp)
                words = model.keys()
        else:
            raise Exception("Unknown Type")

        if lower:
            vectors = {word.lower(): model[word] for word in words}
        else:
            vectors = {word: model[word] for word in words}

        if "UNK" not in vectors:
            unk = np.mean([vectors[word] for word in vectors.keys()], axis=0)
            vectors["UNK"] = unk

        return vectors, len(vectors["UNK"])
