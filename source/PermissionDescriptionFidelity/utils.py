import csv
import os
import re
from collections import Counter

import numpy as np
from gensim.models import KeyedVectors
from gensim.models.wrappers import FastText
from nltk import sent_tokenize, word_tokenize


class Document:
    def __init__(self, doc_id, title, description, permissions):
        self.id = doc_id
        self.title = title
        self.permissions = permissions
        self.description = description 
    
    def __str__(self):
        print(self.title)
        print(self.permissions)



class Permission:
    def __init__(self, permission_type, permission_phrase):
        self.ptype = permission_type
        self.pphrase = permission_phrase
    
    def __str__(self):
        print(self.ptype)
        print(self.pphrase)



class Utils:
    def preprocess(text):
        paragrahps = text.split("\n")
        sentences = []
        for p in paragrahps:
            for s in sent_tokenize(p):
                sentences.append(s)
        return sentences

    def remove_hyperlinks(text):
        regex = r"((https?:\/\/)?[^\s]+\.[^\s]+)"
        text = re.sub(regex, '', text)
        return text

    def to_lower(w, lower):
        return w.lower() if lower else w

    def vocab(file_path, lower=True):
        wordsCount = Counter()
        permissions = []
        distincts_permissions = set()
        with open(file_path) as f:
            reader = csv.reader(f)
            next(reader) # skip header
            for row in reader:
                text = row[1]
                for sentence in Utils.preprocess(text):
                    sentence = Utils.remove_hyperlinks(sentence)
                    for w in word_tokenize(sentence):
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
        return wordsCount.keys(), {w: i for i, w in enumerate(list(wordsCount.keys()))}, permissions

    def read_csv(file_path, w2i, lower=True):
        data = []
        doc_id = 0
        with open(file_path) as f:
            reader = csv.reader(f)
            next(reader) # skip header
            for row in reader:
                doc_id += 1
                title = row[0]
                description = row[1]
                permissions = []
                for p in  row[2].strip().split(","):
                    ptype = Utils.to_lower(p, lower)
                    pphrase = [Utils.to_lower(t, lower) for t in p.split("_")]
                    perm = Permission(ptype, pphrase)
                    permissions.append(perm)

                sentences = []
                for sentence in Utils.preprocess(description):
                    sentence = Utils.remove_hyperlinks(sentence)
                    sentences.append([Utils.to_lower(w, lower) for w in word_tokenize(sentence)])            
                yield Document(doc_id, title, sentences, permissions)
                
    def load_embeddings_file(file_name, embedding_type, lower=True):
        if not os.path.isfile(file_name):
            print(file_name, "does not exist")
            return {}, 0
            
        if embedding_type == "word2vec":
            model = KeyedVectors.load_word2vec_format(file_name, binary=True, unicode_errors="ignore")
            words = model.index2entity
        elif embedding_type == "fasttext":
            model = FastText.load_fasttext_format(file_name)
            words = [w for w in model.wv.vocab]
        else:
            print("Unknown Type")
            return {}, 0

        if lower:
            vectors = {word.lower(): model[word] for word in words}
        else:
            vectors = {word: model[word] for word in words}

        if "UNK" not in vectors:
            unk = np.mean([vectors[word] for word in vectors.keys()], axis=0)
            vectors["UNK"] = unk

        return vectors, len(vectors["UNK"])
