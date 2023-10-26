import spacy
import unidecode
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


def clean(X) -> list[str]:
    result = [
        unidecode.unidecode(token.lemma_.lower()) for token in X 
        if (not token.is_punct) 
        and (not token.is_space) 
        and (not token.is_stop) 
        and len(token) > 1 
    ]
  
    return result

class SpacyTextPreprocessor(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        return self
    
    def transform(Self, X) :
        preproc = spacy.load("fr_core_news_md")
        # handling CamelCase
        default_infixes = list(preproc.Defaults.infixes)
        default_infixes.append('[A-Z][a-z0-9]+')
        infix_regex = spacy.util.compile_infix_regex(default_infixes)
        preproc.tokenizer.infix_finditer = infix_regex.finditer

        docs = preproc.pipe(X.copy())
        prepocessed_texts = []
        for doc in docs:
            prepocessed_texts.append(" ".join(clean(doc)))

        
        return prepocessed_texts

        # return X.copy().apply(lambda x: " ".join(clean(preproc(x))))