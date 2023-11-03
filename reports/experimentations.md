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


# Experimentation sur le modèle de Named Entity Recognition 

## Token preprocessing avec NLTK

Pour cette tâche nous avons extrait des features pour chaqu'un des tokens. Les features retenus dans un premier temps sont toute de type binaires

- **is_capitalized**: si la première lettre est une majuscule
- **is_lemma**: si la lemmatisation du token ne produit pas de changement
- **is_starting_word**: si le token est le premier mots du titre 
- **is_final_word**: si le token est le dernier mots du titre  
- **is_punct**: si le token est de la ponctuation
- **is_stop**: si le token est un stopword contenu dans la liste de nltk


## Prédiction token par token

Notre première approche a été de predire un résultat pour chaque token indépendament de la séquence dans laquelle il se trouve.   
Pour cela nous avons "aplatit" les inputs et outputs utilisé pour l'entrainement.  
Le problème de cette méthode est que l'on se retrouve avec un dataset déséquilibré, ce qui pousse notre modèle à toujour prédir la label et le rend inutilisable pour la 3ème tâche.
Un moyen de contrer ce problème est d'utiliser le paramètre *class_weight* qui permet de changer les poids associé à une classe.
RandomForestClassifier, avec `class_weight={0:1, 1:5}`, ne prédit plus uniquement des 0 mais  les résultats ne sont pas satisfaisant.


## Prédiction séquence par séquence

Nous avons ensuite essayé des modèles capable de faire du *sequences labeling*. Il a fallut appliquer unpadding a notre dataset dans le but d'avoir des séquences de même tailles pour l'entrainement. Pour cela les séquences les plus courtes sont compléter avec une valeur par default. Ensuite le modèle les ignore avec un masque.

### LSTM
Les features retenue sont les même que pour la prédiction token par token.
Malgrès plusieurs essai de paramètres et de régularisation le modèle ne prédit que des 0  
Piste à explorer:
- samples_weights -> pour permettre d'addreser le problème de répartition des label

### CRF
Les features retenue sont les même que pour la prédiction token par token et on y ajoute:
- **word**: le mot complet
- **prefix-2**: les 2 premiers charactères
- **suffix-2**: les 2 derniers charactères
Cela est possible car les CRF prennent des string en entrée, les labels doivent aussi être converti en string

Cette approche est la plus satisfaisante pour l'instant.

Pistes à explorer:
- ajouter des features donnant des informations sur le token précèdent et suivant 
