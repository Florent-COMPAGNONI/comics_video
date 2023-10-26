import numpy as np
import gensim 
import pandas as pd

df = pd.read_csv('/Users/boes/Data/NLP/names_train.csv')
text = df['video_name'].tolist()

training_sentences = [''.join(text)]
model = gensim.models.Word2Vec(training_sentences, vector_size=100)
w2v = {word: model.wv[word] for word in model.wv.index_to_key}

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