class SentenceReport:
    def __init__(self, id, sentence):
        self.app_id = id
        self.sentence = sentence
        self.permissions = {}
        self.preprocessed_sentence = None
        self.prediction_result = None
        self.index_tensor = None


class Reviews:
    def __init__(self, app_id):
        self.app_id = app_id
        self.sentences = []
        self.preprocessed_sentences = []
        self.index_tensors = None
        self.prediction_result = None

class DocumentReport:
    def __init__(self, app_id):
        self.app_id = app_id
        self.permissions = {}
        self.preprocessed_sentences = []
        self.sentences = []
        self.prediction_result = None
        self.index_tensors = None