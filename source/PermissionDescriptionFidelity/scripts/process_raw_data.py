"""This script is used for processing crawled data set"""

import os,sys,inspect

import csv
import os

import langdetect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from utils.nlp_utils import NLPUtils


def process_raw_dataset(file_path, out_file):
    """TODO"""
    with open(file_path) as stream:
        with open(out_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            reader = csv.reader(stream)
            header = next(reader)
            writer.writerow(header)
            for row in reader:
                text = row[1]
                try:
                    sentences = []
                    if langdetect.detect(text) == u'en':
                        for sentence in NLPUtils.sentence_tokenization(text):
                            sentence = NLPUtils.remove_hyperlinks(sentence)
                            sentence = sentence.lower()
                            if sentence:
                                tokens = NLPUtils.word_tokenization(sentence)
                                tokens = [NLPUtils.punctuation_removal(token) for token in tokens]
                                tokens = NLPUtils.stopword_elimination(tokens)
                                tokens = NLPUtils.nonalpha_removal(tokens)
                                if tokens:
                                    sentence = " ".join(tokens)
                                    sentence = sentence.rstrip()
                                    if sentence != "":
                                        sentences.append(sentence.rstrip())
                        writer.writerow([NLPUtils.punctuation_removal(row[0]),
                                         "%%".join(sentences),
                                         "%%".join(row[2].split(",")),
                                         row[3]])
                except Exception:
                    pass

def save_apps_with_given_permission(file_path, included_permission, excluded_permissions_set=None):
    """TODO"""
    file_dir = os.path.dirname(file_path)
    file_name = "{}.csv".format(included_permission.lower())
    out_file = os.path.join(file_dir, file_name)

    with open(file_path) as stream:
        with open(out_file, 'w', newline='') as csvfile:
            reader = csv.reader(stream)
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            header = next(reader)
            writer.writerow(header)
            for row in reader:
                title = row[0]
                text = row[1]
                permissions = row[2]
                link = row[3]

                app_perms = {perm for perm in permissions.split("%%")}
                if included_permission in app_perms:
                    if excluded_permissions_set:
                        if not excluded_permissions_set.intersection(app_perms):
                            writer.writerow([title, text, included_permission, link])
                    else:
                        writer.writerow([title, text, included_permission, link])

if __name__ == "__main__":
    IN_PATH = "/home/huseyin/Desktop/Security/data/small_processed/apps_mini.csv"
    OUT_PATH = "/home/huseyin/Desktop/Security/data/small_processed/apps_mini_processed.csv"
    process_raw_dataset(IN_PATH, OUT_PATH)
    save_apps_with_given_permission(OUT_PATH, "READ_CONTACTS")
    #save_apps_with_given_permission(PATH, "READ_CALENDAR", {"READ_CONTACTS", "RECORD_AUDIO"})
    #save_apps_with_given_permission(PATH, "RECORD_AUDIO", {"READ_CONTACTS", "READ_CALENDAR"})
