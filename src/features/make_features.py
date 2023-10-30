from features.token_features import FlattenTransformer, FlattenTransformer2, TokenFeaturesWithNLTK, ReshapeTransformer, custom_split
import spacy
import pickle

def make_features(df, task):
    y = get_output(df, task)

    X = df["video_name"]

    if task == "is_name":
        X, y = make_ner_features(X, y)
    
    elif task == "find_comic_name":
        X, y = make_comic_name_features(X,y)

    return X, y

def make_comic_name_features(X,y):
    comic_names = y.tolist()
    comic_names_as_tokens = []
    video_name_as_tokens = []
    i = 0
    """nlp = spacy.load("fr_core_news_md")
    for doc in nlp.pipe(X) :
        comic_name_tokens = []
        for token in doc:"""
    for doc in X :
        comic_name_tokens = []
        tokens = custom_split(doc)
        video_name_as_tokens.append(tokens)
        for token in tokens:
            comic_name_tokens.append(1 if str(token) in comic_names[i] else 0)
        comic_names_as_tokens.append(comic_name_tokens)
        i+=1

    with open(r"./models/model_is_comic_video.pkl", 'rb') as f:
         model_is_comic_video = pickle.load(f)
    
    with open(r"./models/model_is_name.pkl", 'rb') as f:
         model_is_name = pickle.load(f)

    X_is_comic_video = X
    X_is_name, y_is_name = make_ner_features(X,y)

    feature_1 = model_is_comic_video.predict(X_is_comic_video)
    feature_2 = model_is_name.predict(X_is_name)
    print(f"Are there 1's in the predicted array ? {1 in feature_2}") 
    feature_2 = ReshapeTransformer().fit(video_name_as_tokens).transform(feature_2)

    # pas encore le bon format pour donner au model 
    features = list(list(zip(feature_1,feature_2)))
    
    print(features)
    return features, comic_names_as_tokens

def make_ner_features(X, y) -> tuple[list[int], list[int]]:
    """
    Extract feature and flatten for RandomForest
    """
    X_features = TokenFeaturesWithNLTK().fit(X).transform(X)
    X, y = FlattenTransformer().fit(X).transform(X_features, y)

    return X, y


def get_output(df, task):
    if task == "is_comic_video":
        y = df["is_comic"]
    elif task == "is_name":
        y = df["is_name"]
    elif task == "find_comic_name":
        y = df["comic_name"]
    else:
        raise ValueError("Unknown task")

    return y
