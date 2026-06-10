import json
import pickle
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA, TruncatedSVD

"""
Contains any useful functions to be used anywhere in our code.
Stuff like saving model as pickle file, or setting up data, etc...

A lot of these helper functions access/save to specific directories. These rely on where
you are running the code from. Folders like pickle_objects/ are in the root directory
of the project, so when running the python script, you NEED to run from that root directory.
So doing something like python3 -m src.execute_scripts.model_setup ... so the saving and loading
is done correctly to how the project is set up.

Always run from root.
"""


def save_model_pickle(model: Any, name: str) -> None:
    with open(f"models/pickle_objects/{name}.pkl", "wb") as file:
        pickle.dump(model, file)


def load_model_pickle(name: str) -> Any:
    with open(f"models/pickle_objects/{name}.pkl", "rb") as file:
        return pickle.load(file)


# the tenseal and pailier models utilise torch tensors for the gradient descent algs
def convert_to_torch_tensors(
    X_train, y_train, X_test, y_test, red_X_train, red_X_test
) -> tuple:
    t_X_train = torch.tensor(X_train).float()
    t_y_train = (
        torch.tensor(y_train).float().unsqueeze(1)
    )  # adds extra dimension [n, 1] so that it corresponds with X when training
    t_X_test = torch.tensor(X_test).float()
    t_y_test = torch.tensor(y_test).float().unsqueeze(1)
    t_red_X_train = torch.tensor(red_X_train).float()
    t_red_X_test = torch.tensor(red_X_test).float()

    return t_X_train, t_y_train, t_X_test, t_y_test, t_red_X_train, t_red_X_test


# pca was saved with n_components=200
def pca_data(X_train, X_test, n_comp) -> tuple:
    pca = PCA(n_components=n_comp)
    red_X_train = pca.fit_transform(X_train)
    red_X_test = pca.transform(
        X_test
    )  # transfrom based off learnt features in train set (we shouldn't learn from test! We can't 'look' into the future)

    return red_X_train, red_X_test


def svd_data(X_train, X_test, n) -> tuple:
    svd = TruncatedSVD(n_components=n)
    red_X_train = svd.fit_transform(X_train)
    red_X_test = svd.transform(X_test)

    return red_X_train, red_X_test


# incase we ever need to run again/change params
def train_pca() -> None:
    pca = PCA(n_components=200)
    save_model_pickle(pca, "pca")


def model_data_path() -> str:
    return "preprocessed_data/model_data"


def reduced_model_data_path() -> str:
    return "preprocessed_data/reduced_model_data"


def load_single_spam_email_df_test(id: int) -> pd.DataFrame:
    test = pd.read_json("../spam_dataset/test.jsonl", lines=True)
    test_spam = test.loc[test["label"] == 1]

    return test_spam.loc[test_spam["message_id"] == id]


# df for pandas dataframe
def load_single_spam_email_df(i: int) -> pd.DataFrame:
    if i > 1000:
        raise ValueError(
            f"There's only 1000 spam examples. You selected {i} - pick <= 1000"
        )

    test = pd.read_json("../spam_dataset/test.jsonl", lines=True)
    test_spam = test.loc[test["label"] == 1]

    return test_spam.iloc[
        [i]
    ]  # iloc selects for actual row index (keep in [[]] to preserve outer list - won't break preprocessing this way)


# for preprocessing single email during demo
def get_indexed_dict() -> dict[str, int]:
    with open("indexed100.json", "r") as file:
        return json.load(file)


def load_single_ham_email(id: int) -> pd.DataFrame:
    test = pd.read_json("dataset/test.jsonl", lines=True)
    test_ham = test.loc[test["label"] == 0]
    # print(test_ham)

    return test_ham.loc[test_ham["message_id"] == id]


# prob don't want to do id. Rathe index in the list (which id doesn't count i don't think)
def print_selected_test_email(index: int) -> None:
    test_emails = pd.read_json("dataset/test.jsonl", lines=True)
    # print(test_emails)
    selected_email = test_emails.loc[index]["text"]

    print(f"Email:\n\n {selected_email}\n")


def randomise(n: int, X, y) -> tuple:
    # these might have to be changed to an np.array
    randomised_X = np.zeros((n, len(X[0])))
    randomised_y = np.zeros((n, 1))
    rand = None
    for i in range(n):
        rand = random.randint(0, 1999)  # 2000 is out of bounds..
        randomised_X[i] = X[rand]
        randomised_y[i] = y[rand]

    return randomised_X, randomised_y, rand


def load_test_data() -> tuple:
    model_data = load_model_pickle(model_data_path())
    reduced_model_data = load_model_pickle(reduced_model_data_path())

    X_test = model_data.get_X_test()  # MAKE SURE TO NOT INCLUDE THESE WHEN TESTING, BY ACCIDENT!!! (yes i spent the past hr debugging this)
    y_test = model_data.get_y_test()
    X_test = X_test.toarray()  # remember these are sparse when loaded
    red_X_test = reduced_model_data.get_red_X_test()

    return X_test, red_X_test, y_test
