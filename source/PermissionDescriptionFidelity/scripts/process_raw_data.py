"""This script is used for processing crawled data set"""
import csv

import langdetect

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
                            if sentence:
                                tokens = NLPUtils.word_tokenization(sentence)
                                tokens = NLPUtils.stopword_elimination(tokens)
                                tokens = [NLPUtils.punctuation_removal(token) for token in tokens]
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

def save_apps_with_given_permission(file_path, included_permission, excluded_permissions_set):
    """TODO"""
    out_file = "{}.csv".format(included_permission)

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
                    if not excluded_permissions_set.intersection(app_perms):
                        writer.writerow([title, text, included_permission, link])




if __name__ == "__main__":

    PATH = "/home/huseyin/Desktop/Security/data/big_processed/processed_big.csv"

    save_apps_with_given_permission(PATH, "READ_CONTACTS", {"READ_CALENDAR", "RECORD_AUDIO"})
    save_apps_with_given_permission(PATH, "READ_CALENDAR", {"READ_CONTACTS", "RECORD_AUDIO"})
    save_apps_with_given_permission(PATH, "RECORD_AUDIO", {"READ_CONTACTS", "READ_CALENDAR"})
