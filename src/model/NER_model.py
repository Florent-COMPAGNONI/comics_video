import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import tensorflow as tf
import sklearn_crfsuite
from sklearn_crfsuite.metrics import sequence_accuracy_score
from features.token_features import  TokenFeaturingAndPadding, TokenFeaturesWithNLTK,custom_split

def remove_padding(sentences,predicted_labels):
    transformed_labels = []
    for index in range(len(predicted_labels)):
        sentence_size = len(custom_split(sentences[index]))
        label = predicted_labels[index][:sentence_size]
        transformed_labels.append(label)
    
    return transformed_labels

class NER_model:
    """
    use RandomForestClassifier and predict token by token (require flatten)
    """
    def __init__(self):
        self.pipeline = Pipeline(
            [('NLTk', TokenFeaturesWithNLTK()),
             ('Padding', TokenFeaturingAndPadding()),
            ("clf", RandomForestClassifier())], 
        )

    def fit(self, X, y):
        #y_padded = tf.keras.preprocessing.sequence.pad_sequences(y, maxlen=30, padding='post', truncating='post', value=0)
        self.pipeline = self.pipeline.fit(X,y)

    def predict(self, X) :
        predicted_labels = self.pipeline.predict(X)
        # remove padding
        predicted_labels = remove_padding(X,predicted_labels)
        return predicted_labels

class NER_model_v2:
    """
    use of crf from sklearn_crfsuite
    """
    def __init__(self, model_file: str = None):
        if model_file:
            with open('crf_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
        else:
            self.model = sklearn_crfsuite.CRF(
                algorithm='lbfgs',
                c1=0.1,
                c2=0.1,
                max_iterations=500,
                all_possible_transitions=True,
            )
    
    def fit(self, X, y):
        y = [
            [str(label) for label in labels] 
            for labels in y
        ]
        train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.2)

        self.model.fit(train_data, train_labels)

        # show accurracy
        predicted_labels = self.model.predict(test_data)
        accuracy = sequence_accuracy_score(test_labels, predicted_labels)
        print(f"Accuracy: {accuracy * 100:.2f}%")
    
    def predict(self, X) -> list[list[int]]:
        predicted_labels = self.model.predict(X)
        # returns int
        predicted_labels = [
            [int(label) for label in labels if label in ['0', '1']] # on exclu les padding
            for labels in predicted_labels
        ]
        return predicted_labels
    
    def evaluate(self, X, y) -> float:
        predicted_labels = self.model.predict(X)
        accuracy = sequence_accuracy_score(y, predicted_labels)
        print(f"Accuracy: {accuracy * 100:.2f}%")

    def save_model(self, filename: str) -> None:
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)