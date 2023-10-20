import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction import DictVectorizer


# download using command: python3 -m spacy download fr_core_news_sm
nlp = spacy.load('fr_core_news_sm')


# Custom transformer to extract token-level features
class TokenFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return [
            {
                'word': token.text,
                'is_capitalized': token.text[0].isupper(),
                'prefix-2': token.text[:2],
                'suffix-2': token.text[-2:]
            } 
            for doc in nlp.pipe(X) for token in doc 
            if not (token.is_punct or token.is_stop)
        ]


# Custom transformer to extract morphological features
class MorphologicalFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return [{'lemma': token.lemma_} for doc in nlp.pipe(X) for token in doc]


# Custom transformer to extract part-of-speech features
class POSTags(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return [{'pos': token.pos_} for doc in nlp.pipe(X) for token in doc]


# Constructing the pre-processing pipeline
def make_features_piline():
    return ('features', FeatureUnion([
        ('token_features', Pipeline([
            ('extractor', TokenFeatures()),
            ('vectorizer', DictVectorizer())
        ])),
        ('morphological_features', Pipeline([
            ('extractor', MorphologicalFeatures()),
            ('vectorizer', DictVectorizer())
        ])),
        ('pos_features', Pipeline([
            ('extractor', POSTags()),
            ('vectorizer', DictVectorizer())
        ])),
    ]))


