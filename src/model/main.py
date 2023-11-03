from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from features.nltk_preprocessing import NltkTextPreprocessor
from model.NER_model import NER_model, NER_model_v2, NER_model_v3

from features.spacy_preprocessing import clean, SpacyTextPreprocessor
from features.embedding import MeanEmbeddingVectorizer

import tensorflow as tf

def make_model():
    return Pipeline(
        [
            # ("nltk_preprocessor", NltkTextPreprocessor()),
            ("spacy_preprocessor", SpacyTextPreprocessor()),
            # ("count_vectorizer", CountVectorizer()),
            ("tfidf_vectorizer", TfidfVectorizer()),
            #("embedding", MeanEmbeddingVectorizer()),
            ("random_forest", RandomForestClassifier()),
        ]
    )


def make_ner_model():
    return NER_model().pipeline

def make_ner_model_v2():
    return NER_model_v2()

def make_ner_model_v3():
    return NER_model_v3()