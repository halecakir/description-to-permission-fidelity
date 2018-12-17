import dynet as dy

from utils import Utils

import random
random.seed(33)

class SimpleModel:
    def __init__(self, vocab, w2i, p2i, options):
        self.model = dy.ParameterCollection()
        self.trainer = dy.AdamTrainer(self.model)
        
        self.w2i = w2i
        self.p2i = p2i
        self.wdims = options.wembedding_dims
        self.ldims = options.lstm_dims
        
        self.wlookup = self.model.add_lookup_parameters((len(w2i), self.wdims)) #PAD, and INITIAL tokens?
        if options.external_embedding is not None:
            ext_embeddings, ext_emb_dim = Utils.load_embeddings_file(options.external_embedding, lower=self.lowerCase, type=options.external_embedding_type)
            assert (ext_emb_dim == self.wdims)
            print("Initializing word embeddings by pre-trained vectors")
            count = 0
            for word in self.vocab:
                if word in ext_embeddings:
                    count += 1
                    self.wlookupphrasep.init_row(self.vocab[word], ext_embeddings[word])
            self.ext_embeddingspphrase = ext_embeddings
            print("Vocab size: pphrase%d; #words having pretrained vectors: %d" % (len(self.vocab), count))
            
        self.sentence_rnn = [dy.SimpleRNNBuilder(1, self.wdims, self.ldims, self.model)] # Try bi-rnn and lstm
        self.permission_rrn = [dy.SimpleRNNBuilder(1, self.wdims, self.ldims, self.model)] # Try bi-rnn and lstm
    
    def cos_similiariy(self, v1, v2):
        from numpy import dot
        from numpy.linalg import norm
        return dot(v1, v2)/(norm(v1)*norm(v2))
    
    def train(self, file_path):
        for doc in Utils.read_csv(file_path, self.w2i, self.p2i):
            if doc.description:
                #TODO: gather all permission encodings and compare them with all sentences of the permission
                rnn_forward = self.permission_rrn[0].initial_state()
                perm_enc_s = []
                for perm  in doc.permissions:
                    for entry in perm.pphrase:
                        vec = self.wlookup[int(self.w2i.get(entry, 0))]
                        rnn_forward = rnn_forward.add_input(vec)
                    perm_enc_s.append(rnn_forward.output().npvalue())
                
                #Sentence encoding
                sentence_enc_s = []
                for sentence in doc.description:
                    rnn_forward = self.sentence_rnn[0].initial_state()
                    for entry  in sentence:
                        vec = self.wlookup[int(self.w2i.get(entry, 0))]
                        rnn_forward = rnn_forward.add_input(vec)
                    try:
                        sentence_enc_s.append(rnn_forward.output().npvalue())
                    except AttributeError:
                        pass
            
            if len(perm_enc_s) > 0 and len(sentence_enc_s) > 0:
                print(self.cos_similiariy(perm_enc_s[0], sentence_enc_s[0]))
                