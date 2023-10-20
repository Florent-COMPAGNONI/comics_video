from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from features.nltk_preprocessing import NltkTextPreprocessor
from model.NER_model import NER_model

def make_model():
    return Pipeline([
        ("nltk_preprocessor", NltkTextPreprocessor() ),
        ("count_vectorizer", CountVectorizer()),
        ("random_forest", RandomForestClassifier()),
    ])


def make_ner_model():
    return NER_model().pipeline