# Requirements
* pip install nltk
* pip install scikit-learn
* pip install spacy
* pip install beautifulsoup4 
* pip install contractions
* pip install Unidecode
* pip install textblob
* pip install pyspellchecker
* python3 -m spacy download fr_core_news_sm


# Experimentation sur le modèle de text classification

## Preprocessing - NLTK
Nous avons céé une classe NltkTextProcessor qui peut être utilisé dans la pipeline SKLearn pour processer le text avant de le passer à un 'vectorizer'. Nous avons exploré les techniques de text processing suivantes :
- Supression des stopwords (stopwords de nltk.corpus)
- Transformer tout le text en miniscule
- L'élimination des accents (package unidecode)
- Supression des numéros (package regex)
- Stemming où on supprime les derniers charactères (package nltk.stem)
- Lemming où on prend en compte le context et réduit le mot à sa racine (package nltk.stem)
- Supression de tous les signes de ponctuation
- Suppression des doubles espaces 

### Résultats pour NLTK Processor & CountVectorizer & RandomForest

* sans pre-processing: 91.20
* avec preprocessing: 90.20
* avec stopwords: 89.6
* avec FrenchStemmer: 88.89
* avec SnowballStemmer: 89.39
* avec SnowballStemmer & stopwords: 88.49
* avec FrenchStemmer & stopwords: 90.19
* avec lemminazation: 89.69
* avec lemminazation & stopwords: 89.58

Nous remarquons que notre processor NLTK n'aide pas à augmenter l'accuracy de notre RandomForest model, au contraire, le score est plus élevé sans processing du texte. 

## Preprocessing - Spacy
Le Spacy Processor qu'on a créé est très lent, mais performe mieux que le NLTK Processor, probablement parce qu'il est mieux adapté à la langue français (avec l'utilisation du modèle pré-entraîné fr_core_news_md). D'abord nous utilisions le modèle fr_core_news_lg (lg = large), mais pour rendre le processor plus rapide, nous avons préféré d'utiliser le format medium (md).

Nous avons rajouté les étapes de processing suivantes :
- supression des stopwords
- supression signes de ponctuation
- suppression des accents
- supression des espaces
- transformation en minuscule
- lemmazation


### Résultats pour Spacy Processor & CountVectorizer & RandomForest 

* Accuracy 91.7 avec tous les étappes de processing :
- supression des stopwords
- supression signes de ponctuation
- suppression des accents
- supression des espaces
- transformation en minuscule
- lemmazation 
- supression des tokens vides

* Accuracy 92.3 avec :
- Supression des accents
- transformation en minuscule
- Lemmatization

### Count Vectorizer vs TF-IDF Vectorizer
* Meilleur score avec TF IDF, mais minime. Résultats sans preprocessing :
    - Accuracy Count Vectorizer : 91.3
    - Accuracy TF-IDF Vectorizer : 91.4

## Création d'une classe qui applique l'embedding par word2Vec
Dans embedding.py nous utilisons gensim.models.Word2Vec() pour entrainer le Word2Vec() sur notre dataset

Accuracy 87.5, donc ne fonctionne pas très bien