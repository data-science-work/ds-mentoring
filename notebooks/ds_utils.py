#! /usr/bin/env python

from collections import defaultdict
import numpy as np

class MeanEmbeddingVectorizer(object):
    def __init__(self, embedding_model, stop_words=[]):
        self.embedding_model = embedding_model
        self.stop_words = stop_words
    
    def fit(self, raw_documents, y=None):
        return self 

    def transform(self, raw_documents):
        return np.array([
            np.mean([self.embedding_model[word] for word in doc.split() 
                     if word not in self.stop_words], axis=0) 
            for doc in raw_documents], np.float32)
    
    def fit_transform(self, raw_documents, y=None):
        return self.transform(raw_documents)

def load_word_embeddings(embedding_path, skip_lines=1, binary=False):
    
    read_style = 'rb' if binary else 'r'
    
    with open(embedding_path, read_style) as embedding_file:
        keys_embds_list = embedding_file.read().splitlines()[skip_lines:]
        dim = len(keys_embds_list[0].split()) - 1
        w_embds = defaultdict(lambda: np.zeros(dim, np.float32))
        for key_embd in keys_embds_list:
            w_v = key_embd.split()
            w_embds[w_v[0].lower().strip()] = np.array(w_v[1:], np.float32)
            
    return w_embds