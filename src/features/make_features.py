from features.token_features import FlattenTransformer, TokenFeaturesWithNLTK


def make_features(df, task):
    y = get_output(df, task)

    X = df["video_name"]

    if task == "is_name":
        return make_ner_features(X, y)

    return X, y


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
