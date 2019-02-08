import csv
import os
import re
from collections import Counter

import numpy as np
import xlrd
from gensim.models import KeyedVectors
from gensim.models.wrappers import FastText


from nltk import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer
import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')
import tkinter


class Application:
    def __init__(self, app_id, dsc_sentences, description, related_permission_doc, tags):
        self.app_id = app_id
        self.dsc_sentences = dsc_sentences
        self.description = description
        self.related_permission_doc = related_permission_doc
        self.tags = tags

    # def __str__(self):
    #     print(self.app_id)
    #     print(self.description)


class DscSentence:
    def __init__(self, sentence, chunk_list, permission_doc, manual_marked, key_based, whyper_tool):
        self.sentence = sentence
        self.chunk_list = chunk_list
        self.permission_doc = permission_doc
        self.manual_marked = manual_marked
        self.key_based = key_based
        self.whyper_tool = whyper_tool

    # def __str__(self):
    #     print(self.sent)


class Document:
    def __init__(self, doc_id, title, description, permissions, tags=None):
        self.id = doc_id
        self.title = title
        self.permissions = permissions
        self.description = description
        self.tags = tags

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


class OverallResultRow:
    def __init__(self, permission, SI, TP, FP, FN, TN, precision_percent, recall_percent, FScore_percent, accuracy_percent):
        self.permission = permission
        self.SI = SI
        self.TP = TP
        self.FP = FP
        self.FN = FN
        self.TN = TN
        self.precision_percent=precision_percent
        self.recall_percent = recall_percent
        self.FScore_percent = FScore_percent
        self.accuracy_percent = accuracy_percent

class Utils:

    @staticmethod
    def read_word_vec(filepath, first_index=0):

        #if not os.path.isfile(filepath):
         #   print("Wiki Vectors File path {} does not exist. Exiting...".format(filepath))
          #  sys.exit()

        word_vec = {}
        with open(filepath) as fp:
            cnt = 0
            for line in fp:
                if cnt < first_index:
                    cnt += 1
                else:
                    #print("line {} contents {}".format(cnt, line))
                    Utils.save_word_vec(line.strip().split(' '), word_vec)
                    cnt += 1
                # if cnt == 5000:
                #     break
        print("input vector length : ")
        print(cnt)
        return word_vec

    @staticmethod
    def save_word_vec(words, word_vec):

        index = 0
        key = ""
        value = []

        for word in words:
            if word != '':
                if index == 0:
                    key = word.lower()
                else:
                    value.append(float(word))
            index += 1

        word_vec[key] = value

    @staticmethod
    def read_whyper_data(input_folder, file_path, file_type, lower, chunk_gram, remove_stop_words):
        wordsCount = Counter()
        permissions = []
        distincts_permissions = set()

        applications = []

        if file_type == "excel":
            handtagged_permissions = ["READ_CALENDAR", "READ_CONTACTS", "RECORD_AUDIO"]
            loc = (input_folder + "/" + file_path)
            wb = xlrd.open_workbook(loc)
            sheet = wb.sheet_by_index(0)

            permission_title = file_path.split("/")[-1].split(".")[0]

            app_id = ""
            app_sentences = []
            app_description = ""
            app_tag = ""

            sharp_count = 0

            if file_path == "Read_Contacts.xls":
                for i in range(sheet.nrows):
                    sentence = sheet.cell_value(i, 0)
                    sentence = str(sentence)
                    if sentence.startswith("#"):
                        if sharp_count != 0:
                            applications.append(Application(app_id, app_sentences, app_description, permission_title, app_tag))
                        elif sharp_count == 0:
                            sharp_count = sharp_count + 1
                        app_id = sentence.split("#")[1]
                        app_sentences = []
                        app_description = ""
                        sentence = sheet.cell_value(i, 1)
                        sentence = str(sentence)
                        app_tag = sentence.split("\\")[2]

                    else:
                        if sharp_count != 0:

                            manual_marked = sheet.cell_value(i, 2)
                            key_based = sheet.cell_value(i, 3)
                            whyper_tool = sheet.cell_value(i, 4)

                            app_sentence = ""
                            sentence = sentence.strip()
                            tokenizer = RegexpTokenizer(r'\w+')
                            for w in tokenizer.tokenize(sentence):
                                wordsCount.update([Utils.to_lower(w, lower)])
                                app_sentence += " " + w

                            app_description = app_description + app_sentence.strip() + ". "
                            sent_chunk_list = Utils.chunker(app_sentence, chunk_gram, remove_stop_words)
                            app_sentences.append(DscSentence(app_sentence.strip(), sent_chunk_list, permission_title, manual_marked, key_based, whyper_tool))

            else:
                for i in range(sheet.nrows):
                    sentence = sheet.cell_value(i, 0)
                    sentence = str(sentence)
                    if sentence.startswith("##"):
                        sharp_count += 1
                        if sharp_count % 2 == 1:
                            if sharp_count != 1:
                                applications.append(Application(app_id, app_sentences, app_description, permission_title, app_tag) )
                            app_id = sentence.split("##")[1]
                            app_sentences = []
                            app_description = ""
                        else:
                            app_tag = sentence.split("\\")[2]

                    else:
                        if sharp_count != 0 and sharp_count % 2 == 0:

                            manual_marked = sheet.cell_value(i, 1)
                            key_based = sheet.cell_value(i, 2)
                            whyper_tool = sheet.cell_value(i, 3)

                            app_sentence = ""
                            sentence = sentence.strip()
                            tokenizer = RegexpTokenizer(r'\w+')
                            for w in tokenizer.tokenize(sentence):
                                wordsCount.update([Utils.to_lower(w, lower)])
                                app_sentence += " " + w

                            app_description = app_description + app_sentence.strip() + ". "
                            sent_chunk_list = Utils.chunker(app_sentence, chunk_gram, remove_stop_words)
                            app_sentences.append(DscSentence(app_sentence.strip(), sent_chunk_list, permission_title, manual_marked, key_based, whyper_tool))

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
        return wordsCount.keys(), {w: i for i, w in enumerate(list(wordsCount.keys()))}, permissions, applications

    @staticmethod
    def chunker(sentence, chunk_gram, remove_stop_words):

        try:
            words = nltk.word_tokenize(sentence)
            tagged = nltk.pos_tag(words)
            chunk_parser = nltk.RegexpParser(chunk_gram)
            chunked = chunk_parser.parse(tagged)

            # chunked.draw()

            # print("------------------------------*****************")
            # for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk' or t.label() == 'CLAUSE'):
            #     print(subtree)

            noun_phrases_list = [' '.join(leaf[0] for leaf in tree.leaves())
                                  for tree in chunked.subtrees()
                                  if tree.label() == 'Chunk']

        except Exception as e:
            print(str(e))

        return noun_phrases_list

    @staticmethod
    def to_lower(w, lower):
        return w.lower() if lower else w

    @staticmethod
    def cos_similiariy(v1, v2):
        from numpy import dot
        from numpy.linalg import norm
        return dot(v1, v2)/(norm(v1)*norm(v2))
