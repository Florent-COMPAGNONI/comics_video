from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from features.dict_features_with_spacy import make_features_piline

class NER_model:
    def __init__(self): # TODO add parameter for model choice 
        self.pipeline = Pipeline([
            make_features_piline(),
            ("model", RandomForestClassifier())
        ])


    # TODO add custom method for fit / transform / predict ?

