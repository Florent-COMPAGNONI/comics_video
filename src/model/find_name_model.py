import pickle
import tensorflow as tf
from features.token_features import CombineFeatures
from features.make_features import make_ner_features, custom_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

def transform_tokens_into_name(sentence,predicted_label):
    name_list = []
    tokens = custom_split(sentence)
    # get the position of the 1's in predicted_labels[index]
    name_positions = [i for i, x in enumerate(predicted_label) if x == 1]
    for pos in name_positions :
        name_list.append(tokens[pos])     

    return " ".join(name_list)

def transform_list_names_into_tokens(list_names, X):
    comic_names_as_tokens = []
    i = 0
    for doc in X :
        comic_name_tokens = []
        tokens = custom_split(doc)
        for token in tokens:
            comic_name_tokens.append(1 if str(token) in list_names[i] else 0)
        comic_names_as_tokens.append(comic_name_tokens)
        i+=1
    return comic_names_as_tokens

def remove_padding(sentences,predicted_labels):
    transformed_labels = []
    for index in range(len(predicted_labels)):
        sentence_size = len(custom_split(sentences[index]))
        label = predicted_labels[index][:sentence_size]
        transformed_labels.append(label)
    
    return transformed_labels

class find_name_model:
    def fit(self, X, y):
        pass

    def predict(self, X):
        with open(r"./models/model_is_comic_video.pkl", 'rb') as f:
            model_is_comic_video = pickle.load(f)
    
        with open(r"./models/model_is_name.pkl", 'rb') as f:
            model_is_name = pickle.load(f)

        feature_1 = model_is_comic_video.predict(X)
        feature_2 = model_is_name.predict(X)

        output = []
        for index in range(len(feature_1)):
            if feature_1[index] == 1 :
                output.append('["' + transform_tokens_into_name(X[index],feature_2[index]) + '"]')
            else :
                output.append('[]')
        return output
    
    def evaluate(self, X, y) -> float:
        comic_names_as_tokens = remove_padding(X,y)
        predicted_labels = self.predict(X)
        predicted_names_as_tokens = transform_list_names_into_tokens(predicted_labels, X)
        size_dataset = len(predicted_labels)
        total_correct = 0
        for index in range(size_dataset):
            if list(comic_names_as_tokens[index]) == predicted_names_as_tokens[index]:
                total_correct+=1
        accuracy = total_correct/size_dataset
        print(f"Accuracy: {accuracy * 100:.2f}%")

    def dump(self, filename_output):
        with open(filename_output, "wb") as f:
            pickle.dump(self, f)


class find_name_model2:
    def __init__(self):
        self.pipeline = Pipeline(
            [('featureCombination', CombineFeatures()),
            ("clf", RandomForestClassifier())]
        )

    def fit(self, X, y):
        self.pipeline = self.pipeline.fit(X,y)

    def predict(self, X):
        predicted_labels = self.pipeline.predict(X)
        # get name out of output
        transformed_predicted_labels = []
        for index in range(len(predicted_labels)):
            transformed_predicted_labels.append(transform_tokens_into_name(X[index],predicted_labels[index]))
        return transformed_predicted_labels

    def dump(self, filename_output):
        with open(filename_output, "wb") as f:
            pickle.dump(self, f)