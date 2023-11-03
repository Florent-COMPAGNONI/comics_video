from typing import Any
import spacy
import re
import string
import numpy as np
from spacy.tokens import Doc
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import tensorflow as tf


def custom_split(s: str) -> list[str]:
    """
    Custom tokenizer used by is_name task
    """
    # Handle special cases : quand on retrouve ': on le remplace par ' "
    s = re.sub(r".':.", "' : ", s)
    # Handle the apostrophe : quand une apostrophe (qui n'est pas au début d'une phrase) est suivi par un charactère, on rajoute un espace entre cette apostrophe et le charactère qui suit 
    s = re.sub(r"([ \w])'([\S]|$)", r"\1' \2", s)
    # Cut into tokens
    tokens = re.findall(r"[\S\u00a0]+|  | $", s)
    return tokens


class TokenFeaturesWithNLTK(BaseEstimator, TransformerMixin):
    """
    Custom transformer to extract token-level features with nltk
    """

    def fit(self, X, y=None):
        self.X = X
        self.wnl = WordNetLemmatizer()
        self.punctuation = string.punctuation
        self.stopwords = stopwords.words("french")
        return self

    def transform(self, X):
        features = []
        for title in tqdm(self.X, desc="Tokens Features Extraction: "):
            splited_title = custom_split(title)
            tokens_data = [
                {
                    "word": token,
                    "is_capitalized": token[0].isupper(),
                    "prefix-2": token[:2],
                    "suffix-2": token[-2:],
                    "is_lemma": token == self.wnl.lemmatize(token), #test si token est égale à la racine
                    "is_starting_word": i == 0,
                    "is_final_word": i == len(splited_title) - 1,
                    "is_punct": token in self.punctuation,
                    "is_stop": token in self.stopwords,
                }
                for i, token in enumerate(splited_title)
            ]
            features.append(tokens_data)
        return features


class TokenFeaturesWithNLTK_v2(BaseEstimator, TransformerMixin):
    """
    Custom transformer to extract token-level features with nltk
    """

    def fit(self, X, y=None):
        self.X = X
        self.wnl = WordNetLemmatizer()
        self.punctuation = string.punctuation
        self.stopwords = stopwords.words("french")
        return self

    def transform(self, X):
        features = []
        for title in tqdm(self.X, desc="Tokens Features Extraction: "):
            splited_title = custom_split(title)
            tokens_data = [
                [
                    1 if token[0].isupper() else 0,                 # is_capitalized
                    1 if token == self.wnl.lemmatize(token) else 0, # is_lemma
                    1 if i == 0 else 0,                             # is_starting_word
                    1 if i == len(splited_title) - 1 else 0,        # is_final_word
                    1 if token in self.punctuation else 0,          # is_punct
                    1 if token in self.stopwords else 0,            # is_stopword
                ]
                for i, token in enumerate(splited_title)
            ]
            features.append(tokens_data)
        return features

class TokenFeaturesWithSpacy(BaseEstimator, TransformerMixin):
    """
    not working
    Custom transformer to extract token-level features with nltk
    download spacy model using command: python -m spacy download fr_core_news_sm
    """

    def fit(self, X, y=None):
        self.X = X

        self.nlp = spacy.load("fr_core_news_sm")
        # prevent splitting
        # self.nlp.tokenizer.token_match = lambda s: re.findall(r"[\S\u00a0]+|  | $", s)
        self.nlp.tokenizer = lambda s: Doc(
            self.nlp.vocab,
            words=custom_split(s),
            spaces=[False] * len(custom_split(s)),
        )
        return self

    def transform(self, X):
        features = []
        for title in self.X:
            splited_title = custom_split(title)
            spacy_words = [token for word in splited_title for token in self.nlp(word)]

            tokens_data = [
                {
                    "word": token.text,
                    "is_capitalized": token.text[0].isupper(),
                    "prefix-2": token.text[:2],
                    "suffix-2": token.text[-2:],
                    "is_lemma": token.text == token[0].lemma_,
                    "is_starting_word": i == 0,
                    "is_final_word": i == len(splited_title) - 1,
                    "is_punct": token.is_punct,
                    "is_stop": token.is_stop,
                }
                for i, token in enumerate(spacy_words)
            ]
            features.append(tokens_data)
        return features


class FlattenTransformer(BaseEstimator, TransformerMixin):
    """
    Flatten the feature dict, to be used to train classic ml model
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_flat = [
            [
                1 if token_features["is_capitalized"] else 0,
                1 if token_features["is_lemma"] else 0,
                1 if token_features["is_starting_word"] else 0,
                1 if token_features["is_final_word"] else 0,
                1 if token_features["is_punct"] else 0,
                1 if token_features["is_stop"] else 0,
            ]
            for sentence_features in X
            for token_features in sentence_features
        ]
        if y is not None:
            y_flat = [label for labels in y for label in labels]
            # print percentage of 1's 
            one_list = [i for i in y_flat if i == 1]
            print(1-(len(one_list)/len(y_flat)))
            return X_flat, y_flat

        return X_flat
    

class TokenFeaturing(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_token_features = []
        for sentence in X:
            sentence_features = []
            for token in sentence :
                token_features = [
                    1 if token["is_capitalized"] else 0,
                    1 if token["is_lemma"] else 0,
                    1 if token["is_starting_word"] else 0,
                    1 if token["is_final_word"] else 0,
                    1 if token["is_punct"] else 0,
                    1 if token["is_stop"] else 0,
                ]
                sentence_features.append(token_features)
            X_token_features.append([token_feature for token in sentence_features for token_feature in token])
            #X_token_features.append(sentence_features)
            X_padded = tf.keras.preprocessing.sequence.pad_sequences(X_token_features, maxlen=30*6, padding='post', truncating='post', value=0)
        
        if y is not None :
            #print(X_padded)
            y_padded = tf.keras.preprocessing.sequence.pad_sequences(y, maxlen=30, padding='post', truncating='post', value=0)
            #print(y_padded)
            return X_padded, y_padded
        
        else :
            return X_padded, None


class ReshapeTransformer(BaseEstimator, TransformerMixin):
    """
    Reshape the sentces after prediction
    """

    def __init__(self):
        self.sentence_lengths = []

    def fit(self, X, y=None):
        # Store the lengths of the sentences for reshaping during transform
        self.sentence_lengths = [len(sentence) for sentence in X]
        return self

    def transform(self, X, y=None):
        X_reshaped = []
        start = 0
        for length in self.sentence_lengths:
            X_reshaped.append(X[start : start + length])
            start += length

        if y is not None:
            start = 0
            y_reshaped = []
            for length in self.sentence_lengths:
                y_reshaped.append(y[start : start + length])
                start += length
            return X_reshaped, y_reshaped

        return X_reshaped


class PaddingTransformer(BaseEstimator, TransformerMixin):
    """
    add padding to input & output to have consistent shape
    work when features are list
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, threshold=18):
        X_padded = []
        y_padded = []
        for title, labels in zip(X, y):
            if len(title) > threshold:
                title = title[:threshold]
                labels = labels[:threshold]

            padded_title = np.pad(
                array=title,
                pad_width=((0, threshold - len(title)), (0, 0)), # ajoute max_len-len(title_splited) à la fin de title splited
                mode="constant", 
                constant_values=-1
            )
            X_padded.append(padded_title)

            padded_labels = np.pad(
                array=labels, 
                pad_width=(0, threshold - len(labels)), 
                mode="constant", 
                constant_values=-1
            )
            y_padded.append(padded_labels)

        return X_padded, y_padded


class PaddingTransformer_V2(BaseEstimator, TransformerMixin):
    """
    add padding to input & output to have consistent shape
    works when features are dict
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, threshold=18):
        PADDING_TOKEN = {
            "word": "<PAD>",
            "is_capitalized": False,
            "prefix-2": "##",
            "suffix-2": "##",
            "lemma": "<PAD>",
            "is_starting_word": False,
            "is_final_word": False,
            "is_punct": False,
            "is_stop": False,
        }
        PADDING_LABEL = -1
        X_padded = [self._pad_list(sentence, threshold, PADDING_TOKEN) for sentence in X]
        y_padded = [self._pad_list(labels, threshold, PADDING_LABEL) for labels in y]

        return X_padded, y_padded

    def _pad_list(self, sentence: list, max_length: int, padding_value: Any):
        while len(sentence) < max_length:
            sentence.append(padding_value)
        return sentence
