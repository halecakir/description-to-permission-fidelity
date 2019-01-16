import random

import dynet as dy
from numpy import inf

from decorators import logging
from utils import Utils

random.seed(33)


class SimpleModel:
    def __init__(self, vocab, w2i, permissions, options):
        self.model = dy.ParameterCollection()
        self.trainer = dy.AdamTrainer(self.model)
        
        self.w2i = w2i
        self.i2w = {w2i[w]:w for w in w2i}
        self.wdims = options.wembedding_dims
        self.ldims = options.lstm_dims
        self.all_permissions = permissions
        self.train_file_type = options.train_file_type
        
        self.wlookup = self.model.add_lookup_parameters((len(w2i), self.wdims)) #PAD, and INITIAL tokens?
        if options.external_embedding is not None:
            ext_embeddings, ext_emb_dim =  Utils.load_embeddings_file(options.external_embedding, options.external_embedding_type)
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
    
    def cosine_proximity(self, pred, gold):
        def l2_normalize(x):
            square_sum = dy.sqrt(dy.bmax(dy.sum_elems(dy.square(x)), np.finfo(float).eps * dy.ones((1))[0]))
            return dy.cdiv(x, square_sum)

        y_true = l2_normalize(pred)
        y_pred = l2_normalize(gold)
        return -dy.sum_elems(dy.cmult(y_true, y_pred))
    
    def cosine_loss(self, pred, gold):
        sn1 = dy.l2_norm(pred)
        sn2 = dy.l2_norm(gold)
        mult = dy.cmult(sn1,sn2)
        dot = dy.dot_product(pred,gold)
        div = dy.cdiv(dot,mult)
        y = dy.scalarInput(2)
        res = dy.cdiv(1-div,y)
        return res
    
    def sentence_permission_sim(self, sentences, perm):
        max_sim = -inf
        max_index = 0
        for index, sentence_enc in enumerate(sentences):
            sim = self.cos_similiariy(sentence_enc, perm)
            if max_sim < sim: 
                max_sim = sim
                max_index = index
        return max_sim, max_index

    def statistics(self, similarities):
        statistics = {}
        for app_id in similarities.keys():
            statistics[app_id] = {"related": { "all" : []},
                                  "unrelated": {"all" : []}}
            max_related, max_unrelated = -inf, -inf
            avg_related, avg_unrelated = 0, 0
            for related_p in similarities[app_id]["related"]:
                statistics[app_id]["related"]["all"].append(related_p[1])

            for unrelated_p in similarities[app_id]["unrelated"]:
                statistics[app_id]["unrelated"]["all"].append(unrelated_p[1])
        return statistics
    
    @logging
    def train(self, documents):
        tagged_loss = 0
        untagged_loss = 0 
        for doc in documents:
            if doc.description:                
                #Sentence encoding
                sentence_enc_s = []
                for sentence,tag in zip(doc.description, doc.tags):
                    sentence_enc_s.append([])
                    if tag == 1 or tag == 2 or tag == 3:
                        for window in sentence:
                            permission_vecs = {}
                            # gather all permission encoding of permissions
                            for perm  in self.all_permissions:
                                rnn_forward = self.permission_rrn[0].initial_state()
                                for entry in perm.pphrase:
                                    vec = self.wlookup[int(self.w2i.get(entry, 0))]
                                    rnn_forward = rnn_forward.add_input(vec)
                                permission_vecs[perm.ptype] = rnn_forward.output()
                                
                            rnn_forward = self.sentence_rnn[0].initial_state()
                            for entry in window:
                                vec = self.wlookup[int(self.w2i.get(entry, 0))]
                                rnn_forward = rnn_forward.add_input(vec)
                            loss = []
                            total_e = 0
                            for perm in self.all_permissions:
                                if perm in doc.permissions:
                                    e = self.cosine_loss(rnn_forward.output(), permission_vecs[perm.ptype])
                                    loss.append(1-e)
                                #else:
                                #    e = self.cosine_loss(rnn_forward.output(), permission_vecs[perm.ptype])
                                #    loss.append(e)
                            loss = dy.esum(loss)
                            tagged_loss += loss.scalar_value()
                            loss.backward()
                            self.trainer.update()
                            dy.renew_cg()
                    
                    elif tag == 0:
                        for window in sentence:
                            permission_vecs = {}
                            # gather all permission encoding of permissions
                            for perm  in self.all_permissions:
                                rnn_forward = self.permission_rrn[0].initial_state()
                                for entry in perm.pphrase:
                                    vec = self.wlookup[int(self.w2i.get(entry, 0))]
                                    rnn_forward = rnn_forward.add_input(vec)
                                permission_vecs[perm.ptype] = rnn_forward.output()
                                
                            rnn_forward = self.sentence_rnn[0].initial_state()
                            for entry in window:
                                vec = self.wlookup[int(self.w2i.get(entry, 0))]
                                rnn_forward = rnn_forward.add_input(vec)
                            loss = []
                            for perm in doc.permissions:
                                loss.append(self.cosine_loss(rnn_forward.output(), permission_vecs[perm.ptype]))
                            loss = dy.esum(loss)
                            untagged_loss += loss.scalar_value()
                            loss.backward()
                            self.trainer.update()
                            dy.renew_cg()
        print("Total loss : {} - Tagged Loss {} - Untagged loss {}".format(tagged_loss+untagged_loss, tagged_loss, untagged_loss))
                            
    @logging
    def test(self, documents, print_mode=False):
        document_permission_similiarities = {}
        permission_vecs = {}
        # gather all permission encoding of permissions
        for perm  in self.all_permissions:
            rnn_forward = self.permission_rrn[0].initial_state()
            for entry in perm.pphrase:
                vec = self.wlookup[int(self.w2i.get(entry, 0))]
                rnn_forward = rnn_forward.add_input(vec)
            permission_vecs[perm.ptype] = rnn_forward.output().npvalue()
        for doc in documents:
            if doc.description:   
                document_permission_similiarities[doc.id] = {"related": [], "unrelated" : []}
                if print_mode:
                    print("\n\nDocument {}".format(doc.id))
                    for sent_id,sent in enumerate(doc.raw_sentences):
                        print("Sentence ID {} : {}".format(sent_id + 1, sent))

                for sent_id, (sentence, tag) in enumerate(zip(doc.description, doc.tags)):
                    sentence_enc_s = []
                    if tag == 1 or tag == 2 or tag == 3:
                        for window in sentence:
                            rnn_forward = self.sentence_rnn[0].initial_state()
                            for entry in window:
                                vec = self.wlookup[int(self.w2i.get(entry, 0))]
                                rnn_forward = rnn_forward.add_input(vec)
                            if rnn_forward.output() is not None:
                                rnn_forward.output().npvalue()
                                sentence_enc_s.append(rnn_forward.output().npvalue())
                            dy.renew_cg()

                        for perm in self.all_permissions:
                            max_sim, max_index = self.sentence_permission_sim(sentence_enc_s, permission_vecs[perm.ptype])
                            if perm in doc.permissions:
                                document_permission_similiarities[doc.id]["related"].append((perm.ptype, max_sim))
                                if print_mode:
                                    print("Sentence ID {} Dependency {}".format(doc.id, sent_id+1, sentence[max_index]))
                            else:
                                document_permission_similiarities[doc.id]["unrelated"].append((perm.ptype, max_sim))
        return document_permission_similiarities
    
    @logging
    def train_test_split(self, file_path, window_size=2):
        documents = []
        for doc in Utils.get_data(file_path,
                                self.w2i, 
                                sequence_type="windowed", 
                                file_type=self.train_file_type, 
                                window_size=window_size, 
                                lower=True):
            documents.append(doc)
        random.shuffle(documents)
        split_point = (3*len(documents))//4
        return documents[:split_point], documents[split_point:]
