import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import tensorflow as tf
import sklearn_crfsuite
from sklearn_crfsuite.metrics import flat_accuracy_score
from features.token_features import TokenFeaturesWithNLTK_v2

class NER_model:
    """
    use RandomForestClassifier and predict token by token (require flatten)
    """
    def __init__(self):
        self.pipeline = Pipeline(
            [("clf", RandomForestClassifier(class_weight={0:1, 1:5}))], # class_weight permet de contrebalancer la faible proportion de 1 dans les label
        )


class NER_model_v2:
    """
    use of LSTM from tensorflow
    so far predict only zeros
    """
    def __init__(self, model_file=None) -> None:
        if model_file:
            self.model = tf.keras.models.load_model(model_file)
        else:
            l1_lambda = 0.01
            l2_lambda = 0.01
            dropout_rate = 0.5
            self.model = tf.keras.Sequential([
                tf.keras.layers.Masking(mask_value=-1., input_shape=(18, 6)), # 18 = threshold choisit au vu de la distribution de la taille des sequences, 6 = nombre de features
                tf.keras.layers.LSTM(
                    units=50, 
                    return_sequences=True,
                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_lambda, l2=l2_lambda),
                    recurrent_regularizer=tf.keras.regularizers.l1_l2(l1=l1_lambda, l2=l2_lambda),
                    dropout=dropout_rate, recurrent_dropout=dropout_rate,
                ),
                tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1_lambda, l2=l2_lambda))
            ])
            self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def fit(self, X, y):
        train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.2)
        train_data = np.array(train_data)
        test_data = np.array(test_data)
        train_labels = np.expand_dims(train_labels, axis=-1)
        test_labels = np.expand_dims(test_labels, axis=-1)

        self.model.fit(train_data, train_labels, epochs=3, batch_size=32, validation_split=0.1)

        loss, accuracy = self.model.evaluate(test_data, test_labels)

        print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    def predict(self, X) -> np.ndarray: # TODO add transform steps ? add post processing to remove padded values ?
        predictions = self.model.predict(np.array(X))
        predictions = np.round(np.squeeze(predictions)) # retire une dimension est  arrondi pour avoir 0 ou 1
        return predictions

    def save_model(self, filename: str) -> None:
        """
        need format .h5, because pickle is not supported by keras
        """
        self.model.save(filename)


class NER_model_v3:
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
        accuracy = flat_accuracy_score(test_labels, predicted_labels)
        print(f"Accuracy: {accuracy * 100:.2f}%")
    
    def predict(self, X): # TODO add transform steps ?
        predicted_labels = self.model.predict(X)
        # returns int
        predicted_labels = [
            [int(label) for label in labels if label in ['0', '1']] # on exclu les padding
            for labels in predicted_labels
        ]
        return predicted_labels

    def save_model(self, filename: str) -> None:
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)