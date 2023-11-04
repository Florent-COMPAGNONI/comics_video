import numpy as np
import gensim 
import pandas as pd
import os

file_path = '/Users/boes/Data/NLP/names_train.csv'

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    text = df['video_name'].tolist()

    training_sentences = [''.join(text)]
    model = gensim.models.Word2Vec(training_sentences, vector_size=100)
    w2v = {word: model.wv[word] for word in model.wv.index_to_key}
else:
    print('The input file to create the embedding does not exist.') 


class MeanEmbeddingVectorizer(object):
    def __init__(self):
        self.word2vec = w2v
        self.dim = 100

    def fit(self, X, y): 
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])