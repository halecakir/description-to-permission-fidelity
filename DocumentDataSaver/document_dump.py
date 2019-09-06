import os
import pickle

import common

seed = 10

class Data:
    def __init__(self):
        self.w2i = None
        self.entries = None
        self.train_entries = None
        self.test_entries = None
        self.ext_embedding = None
        self.reviews = None
        self.predicted_reviews = None

    def load(self, infile):
        with open(infile, "rb") as target:
            self.ext_embeddings, self.entries, self.w2i = pickle.load(target)
    
    def save(self, outfile):
        with open(outfile, "wb") as target:
            pickle.dump([self.ext_embeddings, self.entries, self.w2i], target)
        
    def load_predicted_reviews(self, infile):
        with open(infile, "rb") as target:
            self.predicted_reviews = pickle.load(target)
        for app_id in self.predicted_reviews.keys():
            self.predicted_reviews[app_id].sort(
                key=lambda x: x.prediction_result.item(), reverse=True
            )
            
    def load_reviews(self, infile):
        with open(infile, "rb") as target:
            self.reviews = pickle.load(target)
            

def run(args):
    data = Data()
    data.load(args.saved_data)
    
    DATASET_DIR = os.environ['DATASETS']
    AC_NET_IN = os.path.join(DATASET_DIR, "acnet-data/ACNET_DATASET.csv")
    
    documents = common.__load_row_document_acnet_file(AC_NET_IN, args.stemmer, data.ext_embeddings)
    common.create_document_index_tensors(documents, data.w2i)
    
    #replace sentence entries with documents
    data.entries = documents
    args.saved_data = args.saved_data.replace("sentences", "documents")
    data.save(args.saved_data)
    
    
    
