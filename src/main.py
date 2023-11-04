import click
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.model_selection import cross_val_score
from data.make_dataset import make_dataset
from features.make_features import make_features
from features.token_features import custom_split
from model.main import make_model, make_ner_model, make_ner_model_v2, make_ner_model_v2


@click.group()
def cli():
    pass


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")  # /Users/boes/Data/NLP/names_train.csv
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
def train(task, input_filename, model_dump_filename):
    df = make_dataset(input_filename)
    X, y = make_features(df, task)

    if task == "is_comic_video":
        model = make_model()
    elif task == "is_name" or task == "find_comic_name":
        model = make_ner_model()
    else:
        raise Exception("Invalid task, valid tasks are: is_comic_video, is_name, find_comic_name")
    model.fit(X, y)

    with open(model_dump_filename, "wb") as f:
        pickle.dump(model, f)


@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
@click.option("--output_filename", default="data/processed/prediction.csv", help="Output file for predictions")
def test(task, input_filename, model_dump_filename, output_filename):
    # Read CSV
    df = make_dataset(input_filename)

    # Make features (tokenization, lowercase, stopwords, stemming...)
    X, y = make_features(df, task)

    with open(model_dump_filename, 'rb') as f:
        model= pickle.load(f)
    
    prediction = model.predict(X)
    print(f"Are there 1's in the predicted array ? {1 in prediction}") 
    print(prediction[2])

@click.command()
@click.option("--task", help="Can be is_comic_video, is_name or find_comic_name")
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/model.pkl", help="File to dump model")
def evaluate(task, input_filename, model_dump_filename):
    # Read CSV
    df = make_dataset(input_filename)

    # Make features (tokenization, lowercase, stopwords, stemming...)
    X, y = make_features(df, task)

    # Object with .fit, .predict methods
    if task == "is_comic_video":
        model = make_model()
    elif task == "is_name":
        # """
        # v1 prediction token par token avec RandomForest
        # """
        # model = make_ner_model()
        # model.fit(X[:10_000],y[:10_000])
        
        # predict = model.predict(X[10_000:])
        # print(f"Are there 1's in the predicted array ? {1 in predict}") 
        # print(f"How many ? {list(predict).count(1)} vs {y[10_000:].count(1)} expected")
        # tokens = []
        # for title in df["video_name"].to_numpy():
        #     splitted_title = custom_split(title)
        #     for token in splitted_title:
        #         tokens.append(token)
        # for x, p in zip(tokens, predict):
        #     print(x) if p == 1 else 0
        

        """
        v3 séquences par séquences avec CRF, padding, les features sont sous la forme de dictionnaires
        meilleurs résulats pour l'instant
        """
        model = make_ner_model_v2()
        model.fit(X, y)
        prediction = model.predict(X)
        c = 0
        for p, labels, title in zip(prediction, y, df["video_name"]):
            if 1 in p:
                print(p)
                print(labels)
                print(title)
                c += 1
        print(c)
        model.save_model("models/model_is_name_v3.pkl")

        return # v2 & v3 ne peuvent pas être utilisé avec cross_validation_score

    elif task == "find_comic_name":
        raise Exception("This task is not implemented yet, stay tuned ;)")
    else:
        raise Exception("Invalid task, valid tasks are: is_comic_video, is_name, find_comic_name")

    # with open(model_dump_filename, 'rb') as f:
    #     model = pickle.load(f)

    # Run k-fold cross validation. Print results
    return evaluate_model(model, X, y)


def evaluate_model(model, X, y):
    # Scikit learn has function for cross validation
    list_scores = []
    for i in tqdm(range(1)):
        scores = cross_val_score(model, X, y, scoring="accuracy")
        list_scores.append(np.mean(scores))

    print(f"Got accuracy {100 * np.mean(list_scores)}%")
    return scores


cli.add_command(train)
cli.add_command(test)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
