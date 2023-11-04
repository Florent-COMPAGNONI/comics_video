from features.token_features import (
    FlattenTransformer, 
    PaddingTransformer, 
    TokenFeaturesWithNLTK,
    ReshapeTransformer, 
    TokenFeaturingAndPadding,
    custom_split
)
import pickle
import tensorflow as tf

def make_features(df, task):
    y = get_output(df, task)

    X = df["video_name"].to_numpy()

    if task == "is_name":
        y = tf.keras.preprocessing.sequence.pad_sequences(y, maxlen=30, padding='post', truncating='post', value=0)
        # X, y = make_ner_features(X, y) 
        # X, y = make_ner_features_v2(X, y)"""
    
    elif task == "find_comic_name":
        y = transform_y_find_name(X,y)

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

def transform_y_find_name(X,y):
    comic_names = y.tolist()
    comic_names_as_tokens = []
    for index in range(len(X)) :
        comic_name_tokens = []
        tokens = custom_split(X[index])
        for token in tokens:
            comic_name_tokens.append(1 if str(token) in comic_names[index] else 0)
        comic_names_as_tokens.append(comic_name_tokens)
        
    comic_names_as_tokens_p = tf.keras.preprocessing.sequence.pad_sequences(comic_names_as_tokens, maxlen=30, padding='post', truncating='post', value=0)
    return comic_names_as_tokens_p

def make_ner_features(X, y) -> tuple[list[int], list[int]]:
    """
    Extract feature and flatten for RandomForest
    """
    X_features = TokenFeaturesWithNLTK().fit(X).transform(X)
    X, y = TokenFeaturingAndPadding().fit(X).transform(X_features, y)

    return X, y


def make_ner_features_v2(X, y):
     """
     make ner features for CRF model
     """
     X_features = TokenFeaturesWithNLTK().fit(X).transform(X)
     X, y = PaddingTransformer().fit(X).transform(X_features, y)

     return X, y


def transform_find_comic_name_label(X,predicted_labels):
    list_of_names = []
    for index in range(len(predicted_labels)):
        name_list = []
        tokens = custom_split(X[index])
        # get the position of the 1's in predicted_labels[index]
        name_positions = [i for i, x in enumerate(predicted_labels[index]) if x == 1]
        for pos in name_positions :
            name_list.append(tokens[pos])
        list_of_names.append(" ".join(name_list))

    return list_of_names

def make_comic_name_features(X,y):
    comic_names = y.tolist()
    comic_names_as_tokens = []
    i = 0
    for doc in X :
        comic_name_tokens = []
        tokens = custom_split(doc)
        for token in tokens:
            comic_name_tokens.append(1 if str(token) in comic_names[i] else 0)
        comic_names_as_tokens.append(comic_name_tokens)
        i+=1
    comic_names_as_tokens_p = tf.keras.preprocessing.sequence.pad_sequences(comic_names_as_tokens, maxlen=30, padding='post', truncating='post', value=0)

    with open(r"./models/model_is_comic_video.pkl", 'rb') as f:
         model_is_comic_video = pickle.load(f)
    
    with open(r"./models/model_is_name.pkl", 'rb') as f:
         model_is_name = pickle.load(f)


    feature_1 = model_is_comic_video.predict(X)
    feature_2 = model_is_name.predict(X)

    features = []
    for i in range(len(feature_1)):
        features.append([feature_1[i]] + list(feature_2[i]))
    
    return features, comic_names_as_tokens_p