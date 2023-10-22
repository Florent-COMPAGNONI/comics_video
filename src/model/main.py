from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline

from features.nltk_preprocessing import NltkTextPreprocessor
from features.spacy_preprocessing import clean, SpacyTextPreprocessor

def make_model():
    return Pipeline([
        #("nltk_preprocessor", NltkTextPreprocessor() ),
        ("spacy_preprocessor", SpacyTextPreprocessor() ),
        #("count_vectorizer", CountVectorizer()),
        ("tfidf_vectorizer", TfidfVectorizer()),
        ("random_forest", RandomForestClassifier()),
    ])
