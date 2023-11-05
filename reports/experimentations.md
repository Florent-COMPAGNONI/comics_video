# Requirements
* pip install nltk
* pip install scikit-learn
* pip install spacy
* pip install beautifulsoup4 
* pip install contractions
* pip install Unidecode
* pip install textblob
* pip install pyspellchecker
* pip install sklearn_crfsuite
* python3 -m spacy download fr_core_news_md/fr_core_news_sm
* TensorFlow install

# Experimentation sur le modèle de text classification : is_comic_video

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


# Experimentation sur le modèle de Named Entity Recognition : is_name

2 lignes du dataset nous posait problème à cause du split des tokens qui était différent dans nos étapes de préprocessing et qui résultaient dans un nombre de tokens différents et donc une sortie différent pour les labels. Nous n'avons pas réussi à résoudre ce problème avec notre fonction custom_split(), donc pour tester nos prochains modèles, il faut modifier ces deux lignes (ou les supprimer) : 

* Magali Le Huche : Comment dessiner 'Paco' --> Nous avons rajouté un 0 à la fin. Résultat : [1, 1, 1, 0, 0, 0, 0, 1, 0]
* Geoffroy Roux de Bézieux : "On ne peut pas payer collectivement un 'quoi qu'il en coûte'permanent"  --> Nous avons enlevé un 0 à la fin. Résultat : [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

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
Le problème de cette méthode est que l'on se retrouve avec un dataset déséquilibré, ce qui pousse notre modèle à toujour prédir la label 0 et le rend inutilisable pour la 3ème tâche.
Un moyen de contrer ce problème est d'utiliser le paramètre *class_weight* qui permet de changer les poids associé à une classe.
RandomForestClassifier, avec `class_weight={0:1, 1:5}`, ne prédit plus uniquement des 0 mais  les résultats ne sont pas satisfaisant.


## Prédiction séquence par séquence

Nous avons ensuite essayé des modèles capable de faire du *sequences labeling*. Il a fallut appliquer un padding a notre dataset dans le but d'avoir des séquences de même tailles pour l'entrainement. Pour cela les séquences les plus courtes sont compléter avec une valeur par default. 

### RandomForest (NER_model)
Les features retenue sont les même que pour la prédiction token par token (classe TokenFeaturesWithNLTK), mais nous avons rajouté un padding de taille 30 (car maximum taille de titre dans dataset est 21, et nous supposons que la taille d'un titre ne dépasse pas les 30 tokens). Après padding, on obtient une liste de 2 dimensions par titre, où la première dimension est de taille 30 et la 2ème de taille 6 (six features par token). Vu que le RandomForest de SKLearn exige une liste de dimension 1 par entrée, nous avons flatten les features pour tous les tokens dans un titre, donc par titre on a une liste de token features (qui ne sont plus séparés par token). Toutes ces étappes de préprocessing sent comprises dans la pipeline à l'initialisation du model 'NER_model'

Pour avoir des labels à prédire qui ont toutes la même taille, on applique également un padding de taille 30 sur les données de la colonne is_name (implémenté dans la fonction 'make_features', pour que dans la fonction predict du modèle Random Forest de SKLearn les labels transformées sont pris en compte pendant la cross validation)

Dans la methode 'predict' de notre classe NER_model, nous transformons la sortie de la fonction predict propre au model pour enlever le padding et sortir le même format de donnée qui se trouve dans la colonne is_name de notre dataset (comme ça notre le fichier output de notre fonction test contiendra le bon format direct)

Nous atteignons une accuracy de 76% en moyen avec ces étapes de préprocessing et le Random Forest model. Nous pensons que ce score médiocre est lié aux erreurs que nous avons détectées dans le dataset : à partir de la ligne 300 (environ) is_name est très souvent un tableau de 0 malgrès qu'il s'agit d'une vidéo comique et qu'il y a des noms présent de le titre (et parfois aussi dans la colonne comic_name). Exemples : Le questionnaire Jupiproust de Gérard Garouste - Le questionnaire Jupiproust, Le questionnaire Jupiproust d'Alexis Rosenfeld, Le karcher de Valérie Pécresse - Le Sketch Par Jupiter, Les voisins de mes voisins sont mes voisins - La chronique de Léo Karmann, Bloqué - Le Sketch Par Jupiter avec Gringe,... 

#### Exploration features (dans TokenFeaturingAndPadding)
1. Le premier feature logique est 'is_cap' (si le mot commence avec une majuscule : des noms commencent avec des majuscules). En utilisant seulement ce feauture, on atteint une accuracy de 72%
2. En rajoutant le feautre 'is_starting' (si le mot est positioné au début de la phrase : ceci pourrait être une indice également) : Accuracy de 73% (on monte de 1%)
3. En rajoutant le feauture 'is_final' (si le mot est positioné à la fin d'une phrase : souvent les noms se retrouvent à la fin) : Accuracy de 73,3% (on monte de 0.3%)
4. En rajoutant le feature is_stop (si le mot est un stopwords : les noms ne peuvent pas être des stopwords) : On atteint une accuracy de 76% (on monte de environ 3%), logique vu que les stopwords sont éliminé, car le model va comprendre qu'il ne s'agit jamais des noms
5. On teste de rajouter le feature qui détecte les signe de ponctuation(is_punct) ce qui pourrait élimener plus de tokens, mais bizarrement ceci fait légèrement déscendre l'accuracy de 0.5% (on atteint 75.5%). On décide donc de ne pas utiliser ce feature
6. En rajoutant le feature qui vérifie si le token est un mot "racine" (is_lemme), l'accuracy déscend un tout petit peu : On atteint un accuracy de 75.8%. On n'utilisera pas non plus ce feature pour la prédiction
7. En rajoutant le feature qui vérifie si le token précédent est un stopword (prev_is_stop) on ne gagne pas en accuracy. On atteint toujours 76%
8. En rajoutant le feature qui vérifie si le token précédent est une signe de ponctuation (prev_is_punct) notre accuracy augmente de nouveau et on atteint une accuracy de 76,3%

### CRF (NER_model_v2)
Nous avons testé un deuxième model pour la tâche is_name.
Les features retenue sont les même que pour la prédiction token par token et on y ajoute:
- **word**: le mot complet
- **prefix-2**: les 2 premiers charactères
- **suffix-2**: les 2 derniers charactères
Cela est possible car les CRF prennent des string en entrée, les labels doivent aussi être convertis en string

premier résultat, le modèle est autour de 80% d'accuracy

Pistes à explorer:
- ajouter des features donnant des informations sur le token précèdent et suivant 
- ajouter le POS tagging en features

# Experimentation sur le modèle pour la tâche find_comic_name

Pour cette tâche nous avons testé deux approches différentes. Dans un premier temps nous avons créé un notre propre simple modèle  avec des règle if else se basant sur les sorties des des tâches is_comic_video et is_name. Dans un deuxième temps nous avons utilisés les sorties des tâches is_comic_video et is_name comme entrée pour un Random Forest model.

Dans make_features nous transformer les labels de la colonne comic_name en vecteurs avec des 1 où il y a des noms pour des titres comiques et des vecteurs avec que des 0 pour les titres non comique. 

### Tests
Notre modèle simple if else a une meilleure accuracy que celle qui passe par le Random Forest (93% vs 90%).
Nous avons aussi testé les 2 modelès pour is_name (crf & RandomForest) comme entrée de nos modèles find_name et même si l'accuracy de CRF était plus élevé pour la tâche is_name, en l'utilisant pour créer les features d'entrée pour le model find_name nous avons une moins bonne score d'accuracy avec le CRF par rapport au Random Forest, et nous avons donc choisi de laisser le NER_model dans les fonctions train, test et evaluate du main pour la tâche is_name, et nous utilisons donc le pickle model_is_name (qui correspond au modèle entrainé avec Random Forest) pour créer l'input de nos modelès find_name.