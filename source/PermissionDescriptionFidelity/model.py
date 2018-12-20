import dynet as dy

from utils import Utils

from numpy import inf
import random
random.seed(33)

class SimpleModel:
    def __init__(self, vocab, w2i, permissions, options):
        self.model = dy.ParameterCollection()
        self.trainer = dy.AdamTrainer(self.model)
        
        self.w2i = w2i
        self.wdims = options.wembedding_dims
        self.ldims = options.lstm_dims
        self.all_permissions = permissions
        
        self.wlookup = self.model.add_lookup_parameters((len(w2i), self.wdims)) #PAD, and INITIAL tokens?
        if options.external_embedding is not None:
            ext_embeddings, ext_emb_dim = Utils.load_embeddings_file(options.external_embedding, options.external_embedding_type)
            assert (ext_emb_dim == self.wdims)
            print("Initializing word embeddings by pre-trained vectors")
            count = 0
            for word in self.w2i:
                if word in ext_embeddings:
                    count += 1
                    self.wlookup.init_row(self.w2i[word], ext_embeddings[word])
            self.ext_embeddings = ext_embeddings
            print("Vocab size: %d; #words having pretrained vectors: %d" % (len(self.w2i), count))
            
        self.sentence_rnn = [dy.SimpleRNNBuilder(1, self.wdims, self.ldims, self.model)] # Try bi-rnn and lstm
        self.permission_rrn = [dy.SimpleRNNBuilder(1, self.wdims, self.ldims, self.model)] # Try bi-rnn and lstm
    
    def cos_similiariy(self, v1, v2):
        from numpy import dot
        from numpy.linalg import norm
        return dot(v1, v2)/(norm(v1)*norm(v2))
    
    def description_permission_sim_w_max(self, sentences, perm):
        max_sim = -inf
        for sentence_enc in sentences:
            sim = self.cos_similiariy(sentence_enc, perm)
            if max_sim < sim: max_sim = sim
        return max_sim

    def statistics(self, similarities):
        statistics = {}
        for app_id in similarities.keys():
            statistics[app_id] = {"related": {"max" : None, "avg" : None, "all" : []},
                                  "unrelated": {"max" : None, "avg" : None, "all" : []}}

            max_related, max_unrelated = -inf, -inf
            avg_related, avg_unrelated = 0, 0
            for related_p in similarities[app_id]["related"]:
                if max_related < related_p[1]: 
                    max_related = related_p[1]
                avg_related += related_p[1]
                statistics[app_id]["related"]["all"].append(related_p[1])

            for unrelated_p in similarities[app_id]["unrelated"]:
                if max_unrelated < unrelated_p[1]: 
                    max_unrelated = unrelated_p[1]
                avg_unrelated += unrelated_p[1]
                statistics[app_id]["unrelated"]["all"].append(unrelated_p[1])

            statistics[app_id]["related"]["max"] = max_related
            statistics[app_id]["unrelated"]["max"] = max_unrelated
            statistics[app_id]["related"]["avg"] = avg_related / len(similarities[app_id]["related"])
            statistics[app_id]["unrelated"]["avg"] = avg_unrelated / len(similarities[app_id]["unrelated"])
        return statistics

    def train(self, file_path):
        document_permission_similiarities = {}
        permission_vecs = {}
        # gather all permission encoding of permissions
        for perm  in self.all_permissions:
            rnn_forward = self.permission_rrn[0].initial_state()
            for entry in perm.pphrase:
                vec = self.wlookup[int(self.w2i.get(entry, 0))]
                rnn_forward = rnn_forward.add_input(vec)
            permission_vecs[perm.ptype] = rnn_forward.output().npvalue()
            dy.renew_cg()

        for doc in Utils.read_csv(file_path, self.w2i):
            if doc.description:                
                #Sentence encoding
                sentence_enc_s = []
                for sentence in doc.description:
                    rnn_forward = self.sentence_rnn[0].initial_state()
                    for entry in sentence:
                        vec = self.wlookup[int(self.w2i.get(entry, 0))]
                        rnn_forward = rnn_forward.add_input(vec)
                    if rnn_forward.output() is not None:
                        sentence_enc_s.append(rnn_forward.output().npvalue())
                    dy.renew_cg()
                    
                document_permission_similiarities[doc.id] = {"related": [], "unrelated" : []}
                app_permissions = set()
                for related_p in doc.permissions:
                    sim = self.description_permission_sim_w_max(sentence_enc_s, permission_vecs[related_p.ptype])
                    document_permission_similiarities[doc.id]["related"].append((related_p.ptype, sim))
                    app_permissions.add(related_p.ptype)
                for unrelated_p in self.all_permissions:
                    if unrelated_p.ptype not in app_permissions:
                        sim = self.description_permission_sim_w_max(sentence_enc_s, permission_vecs[unrelated_p.ptype])
                        document_permission_similiarities[doc.id]["unrelated"].append((unrelated_p.ptype, sim))
                
        return document_permission_similiarities

                    
                   
                    
                        
                    
                