import pandas as pd
from ast import literal_eval


def make_dataset(filename):
    df = pd.read_csv(filename)
    df["is_name"] = df["is_name"].apply(literal_eval)
    return df