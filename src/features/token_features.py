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
    token features are represented in a dict
    """
    def fit(self, X, y=None):
        self.wnl = WordNetLemmatizer()
        self.punctuation = string.punctuation
        self.stopwords = stopwords.words("french")
        return self

    def transform(self, X) -> list[list[dict]]:
        features = []
        for title in tqdm(X, desc="Tokens Features Extraction: "):
            splited_title = custom_split(title)
            title_length = len(splited_title)
            tokens_data = []
            for i, token in enumerate(splited_title):
                token_features = {
                    "word": token,
                    "is_capitalized": token[0].isupper(),
                    "prefix-2": token[:2],
                    "suffix-2": token[-2:],
                    "is_lemma": token == self.wnl.lemmatize(token), #test si token est égale à la racine
                    "is_starting_word": i == 0,
                    "is_final_word": i == len(splited_title) - 1,
                    "is_punct": token in self.punctuation,
                    "is_stop": token in self.stopwords,
                    # Features for the previous token (if it exists)
                    "prev_word": splited_title[i - 1] if i > 0 else "<START>",
                    "prev_is_capitalized": splited_title[i - 1][0].isupper() if i > 0 else "<START>",
                    "prev_prefix-2": splited_title[i - 1][:2] if i > 0 else "<START>",
                    "prev_suffix-2": splited_title[i - 1][-2:] if i > 0 else "<START>",
                    "prev_is_lemma": token == self.wnl.lemmatize(token) if i > 0 else "<START>",
                    "prev_is_starting_word": i == 0  if i > 0 else "<START>",
                    "prev_is_final_word": i == len(splited_title) - 1  if i > 0 else "<START>",
                    "prev_is_punct": token in self.punctuation if i > 0 else "<START>",
                    "prev_is_stop": token in self.stopwords if i > 0 else "<START>",
                    # # Features for the next token (if it exists)
                    "next_word": splited_title[i + 1] if i < title_length - 1 else "<END>",
                    "next_is_capitalized": splited_title[i + 1][0].isupper() if i < title_length - 1 else "<END>",
                    "next_prefix-2": splited_title[i + 1][:2] if i < title_length - 1 else "<END>",
                    "next_suffix-2": splited_title[i + 1][-2:] if i < title_length - 1 else "<END>",
                    "is_lemma": token == self.wnl.lemmatize(token) if i < title_length - 1 else "<END>",
                    "is_starting_word": i == 0  if i < title_length - 1 else "<END>",
                    "is_final_word": i == len(splited_title) - 1 if i < title_length - 1 else "<END>",
                    "is_punct": token in self.punctuation if i < title_length - 1 else "<END>",
                    "is_stop": token in self.stopwords if i < title_length - 1 else "<END>",
                }
                tokens_data.append(token_features)
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
    works when features are dict
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, threshold=18) -> list[list[dict]]:
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

    def _pad_list(self, sentence: list, max_length: int, padding_value: Any) -> list:
        while len(sentence) < max_length:
            sentence.append(padding_value)
        return sentence
