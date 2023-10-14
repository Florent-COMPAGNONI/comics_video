from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from features.nltk_preprocessing import NltkTextPreprocessor

def make_model():
    return Pipeline([
        ("nltk_preprocessor", NltkTextPreprocessor() ),
        ("count_vectorizer", CountVectorizer()),
        ("random_forest", RandomForestClassifier()),
    ])
