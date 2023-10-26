import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import FrenchStemmer, SnowballStemmer

from bs4 import BeautifulSoup
from unidecode import unidecode
import re
import string
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class NltkTextPreprocessor(TransformerMixin, BaseEstimator):
  def __init__(self):
    pass

  def fit(self, X, y):
    return self

  def transform(self, X):
    txt_preproc = NltkPreprocessingSteps(X.copy())
    processed_text = \
            txt_preproc \
            .to_lower()\
            .replace_diacritics()\
            .remove_numbers()\
            .remove_double_spaces()\
            .remove_all_punctuations()\
            .remove_stopwords()\
            .stemming()\
            .lemming()\
            .get_processed_text()

    return processed_text

def download_if_non_existent(res_path, res_name):
  try:
    nltk.data.find(res_path)
  except LookupError:
    #print(f'resource {res_path} not found. Downloading now...')
    nltk.download(res_name)

class NltkPreprocessingSteps:
  def __init__(self, X):
    self.X = X
    download_if_non_existent('corpora/stopwords', 'stopwords')
    download_if_non_existent('tokenizers/punkt', 'punkt')
    download_if_non_existent('taggers/averaged_perceptron_tagger',
                             'averaged_perceptron_tagger')
    download_if_non_existent('corpora/wordnet', 'wordnet')
    download_if_non_existent('corpora/omw-1.4', 'omw-1.4')

    self.sw_nltk = stopwords.words('french')
  
  #permet d’éliminer tous les accents
  def replace_diacritics(self):
    self.X = self.X.apply(
            lambda x: unidecode(x, errors="preserve"))
    return self

  def to_lower(self):
    self.X = self.X.str.lower()
    return self

  def remove_numbers(self):
    self.X = self.X.apply(lambda x: re.sub(r'\d+', '', x))
    return self

  def replace_dots_with_spaces(self):
    self.X = self.X.apply(lambda x: re.sub("[.]", " ", x))
    return self

  def remove_all_punctuations(self):
    self.X = self.X.apply(lambda x: re.sub('[%s]' %
                          re.escape(string.punctuation), '' , x))
    return self

  def remove_double_spaces(self):
    self.X = self.X.apply(lambda x: re.sub(' +', ' ', x))
    return self

  def remove_stopwords(self):
    # remove stop words from token list in each column
    self.X = self.X.apply(
            lambda x: " ".join([ word for word in x.split() 
                     if word not in self.sw_nltk]) )
    return self

  def stemming(self):
    ps = FrenchStemmer()
    # ps = SnowballStemmer(language='french')
    self.X = self.X.apply(lambda x: " ".join([
      ps.stem(word) 
      for word in word_tokenize(x)]))
    return self
  
  def lemming(self):
    wnl = WordNetLemmatizer()
    self.X = self.X.apply(lambda x: " ".join([
      wnl.lemmatize(word) 
      for word in word_tokenize(x)]))
    return self

  def get_processed_text(self):
    return self.X
