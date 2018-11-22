# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class TfidfModel:
    def __init__(self, opts):
        pass

    @staticmethod
    def get_doc_idfs(cnts):
        binary = (cnts > 0).astype(int)
        freqs = np.array(binary.sum(0)).squeeze()
        idfs = np.log((cnts.shape[0] - freqs + 0.5) / (freqs + 0.5))
        idfs[idfs < 0] = 0
        return idfs

    @staticmethod
    def matrix_norm(tfidf_matrix):
        norm = 1.0 / (np.sqrt(np.array(tfidf_matrix.multiply(tfidf_matrix).sum(axis=1)).squeeze()) + 1e-8)
        norm_matrix = sp.diags(norm, 0)
        norm_tfidf_matrix = norm_matrix.dot(tfidf_matrix)
        return norm_tfidf_matrix

    @staticmethod
    def get_sim_matrix(tfidf_matrix):
        return tfidf_matrix.dot(tfidf_matrix.T)

    def sklearn_build(self, corpus, fields):
        vectorizer_tfidf = TfidfVectorizer(analyzer='char', ngram_range=(1, 2), min_df=1, norm='l2')
        content = []
        for idx, sample in enumerate(corpus):
            for field in fields:
                content.append(' '.join(sample[field]))
        # print(content)
        vectorizer_tfidf.fit(content)
        return vectorizer_tfidf
        
    def compute_sim(self, vectorizer_tfidf, corpus):
        q1_corpus = [' '.join(t['q1_tokens']) for t in corpus]
        q2_corpus = [' '.join(t['q2_tokens']) for t in corpus]
        truth = [t['match'] for t in corpus]
        q1_vectors = vectorizer_tfidf.transform(q1_corpus)
        q2_vectors = vectorizer_tfidf.transform(q2_corpus)
        sim_score = q1_vectors.dot(q2_vectors.T).diagonal()
        return sim_score, truth
