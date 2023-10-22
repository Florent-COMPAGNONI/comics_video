import spacy
import unidecode
from sklearn.base import BaseEstimator, TransformerMixin

def clean(X) -> list[str]:
    preproc = spacy.load("fr_core_news_lg")

    # handling CamelCase
    default_infixes = list(preproc.Defaults.infixes)
    default_infixes.append('[A-Z][a-z0-9]+')
    infix_regex = spacy.util.compile_infix_regex(default_infixes)
    preproc.tokenizer.infix_finditer = infix_regex.finditer

    txt_preproc = preproc(X)

    return [
        unidecode.unidecode(token.lemma_.lower()) for token in txt_preproc 
        if (not token.is_punct) 
        and (not token.is_space) 
        and (not token.like_url) 
        and (not token.is_stop) 
        and len(token) > 1 
    ]

class SpacyTextPreprocessor(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        return self
    
    def transform(Self, X) :
        return X.copy().apply(lambda x: " ".join(clean(x)))