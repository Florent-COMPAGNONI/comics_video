from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


class NER_model:
    def __init__(self):  # TODO add parameter for model choice
        self.pipeline = Pipeline(
            [("clf", RandomForestClassifier())],
        )
