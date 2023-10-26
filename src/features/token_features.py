import spacy
import stanza
import re
import string
from spacy.tokens import Doc
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm


# download using command: python -c "import stanza; stanza.download('fr')"
# nlp_stanza = stanza.Pipeline(lang="fr")


def custom_split(s: str) -> list[str]:
    """
    Custom tokenizer used by is_name task
    """
    # Handle special cases
    s = re.sub(r".':.", "' : ", s)
    # Handle the apostrophe
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
                    "is_lemma": token == self.wnl.lemmatize(token),
                    "is_starting_word": i == 0,
                    "is_final_word": i == len(splited_title) - 1,
                    "is_punct": token in self.punctuation,
                    "is_stop": token in self.stopwords,
                }
                for i, token in enumerate(splited_title)
            ]
            features.append(tokens_data)
        return features


class TokenFeaturesWithSpacy(BaseEstimator, TransformerMixin):
    """
    Custom transformer to extract token-level features with nltk, not working yet
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
            return X_flat, y_flat

        return X_flat


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