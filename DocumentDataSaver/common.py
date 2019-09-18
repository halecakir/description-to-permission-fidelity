import torch
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


class SentenceReport:
    def __init__(self, id, sentence):
        self.app_id = id
        self.sentence = sentence
        self.permissions = {}
        self.preprocessed_sentence = None
        self.prediction_result = None
        self.index_tensor = None


class Review:
    def __init__(self, sentence, score):
        self.sentence = sentence
        self.preprocessed_sentence = None
        self.score = score
        self.index_tensor = None
        self.prediction_result = None


def __load_row_document_acnet_file(infile, stemmer, embeddings):
    print("Loading row {} ".format(infile))
    # read training data
    print("Reading Train Sentences")
    tagged_train_file = pd.read_csv(infile)
    documents = []
    acnet_map = {
        "RECORD_AUDIO": "MICROPHONE",
        "READ_CONTACTS": "CONTACTS",
        "READ_CALENDAR": "CALENDAR",
        "ACCESS_FINE_LOCATION": "LOCATION",
        "CAMERA": "CAMERA",
        "READ_SMS": "SMS",
        "READ_CALL_LOGS": "CALL_LOG",
        "CALL_PHONE": "PHONE",
        "WRITE_SETTINGS": "SETTINGS",
        "GET_TASKS": "TASKS",
    }

    for idx, row in tagged_train_file.iterrows():
        app_id = row["app_id"]
        sentence = row["sentence"]

        if documents == []:  # if it is the first document
            documents.append(DocumentReport(app_id))
        elif documents[-1].app_id != app_id:  # if it is a new document
            documents.append(DocumentReport(app_id))

        for permission in acnet_map:
            if (
                permission not in documents[-1].permissions
                or row[acnet_map[permission]] == 1
            ):
                documents[-1].permissions[permission] = row[acnet_map[permission]]

        documents[-1].sentences.append(sentence)
        preprocessed = NLPUtils.preprocess_sentence(sentence, stemmer)
        documents[-1].preprocessed_sentences.append(
            [word for word in preprocessed if word in embeddings]
        )

    print("Loading completed")
    return documents


def create_document_index_tensors(documents, w2i):
    def get_tensor(sequence, w2i):
        index_tensor = torch.zeros((1, len(sequence)), dtype=torch.long)
        for idx, word in enumerate(sequence):
            index_tensor[0][idx] = w2i[word]
        return index_tensor

    for document in documents:
        document.index_tensors = []
        for preprocessed_sentence in document.preprocessed_sentences:
            index_tensor = get_tensor(preprocessed_sentence, w2i)
            document.index_tensors.append(index_tensor)
