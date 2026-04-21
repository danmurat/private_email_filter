import pickle
import torch
import pandas as pd
import json
import tenseal as ts
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

"""
Contains any useful functions to be used anywhere in our code.
Stuff like saving model as pickle file, or setting up data, etc...
"""

def saveModelPickle(model, name):
    with open(f"pickle_objects/{name}.pkl", "wb") as file:
        pickle.dump(model, file)

def loadModelPickle(name):
    with open(f"pickle_objects/{name}.pkl", "rb") as file:
        return pickle.load(file)

def convertToTorchTensors(X_train, y_train, X_test, y_test, red_X_train, red_X_test) -> tuple:
    t_X_train = torch.tensor(X_train).float()
    t_y_train = torch.tensor(y_train).float().unsqueeze(1) # adds extra dimension [n, 1] so that it corresponds with X when training
    t_X_test = torch.tensor(X_test).float()
    t_y_test = torch.tensor(y_test).float().unsqueeze(1)
    t_red_X_train = torch.tensor(red_X_train).float()
    t_red_X_test = torch.tensor(red_X_test).float()

    return t_X_train, t_y_train, t_X_test, t_y_test, t_red_X_train, t_red_X_test

# pca was saved with n_components=200
def pca_data(X_train, X_test, n_comp) -> tuple:
    pca = PCA(n_components=n_comp)
    red_X_train = pca.fit_transform(X_train)
    red_X_test = pca.transform(X_test) # transfrom based off learnt features in train set (we shouldn't learn from test! We can't 'look' into the future)

    return red_X_train, red_X_test

def svd_data(X_train, X_test, n) -> tuple:
    svd = TruncatedSVD(n_components=n)
    red_X_train = svd.fit_transform(X_train)
    red_X_test = svd.transform(X_test)

    return red_X_train, red_X_test


# incase we ever need to run again/change params
def train_pca():
    pca = PCA(n_components=200)
    saveModelPickle(pca, "pca")


def model_data_path():
    return "preprocessed_data/model_data"

def reduced_model_data_path():
    return "preprocessed_data/reduced_model_data"


def load_single_spam_email_df_test(id):
    test = pd.read_json("../spam_dataset/test.jsonl", lines=True)
    test_spam = test.loc[test["label"] == 1]

    return test_spam.loc[test_spam["message_id"] == id]

# df for pandas dataframe
def load_single_spam_email_df(i):
    if i > 1000:
        raise ValueError(f"There's only 1000 spam examples. You selected {i} - pick <= 1000")

    test = pd.read_json("../spam_dataset/test.jsonl", lines=True)
    test_spam = test.loc[test["label"] == 1]

    return test_spam.iloc[[i]] # iloc selects for actual row index (keep in [[]] to preserve outer list - won't break preprocessing this way)


# for preprocessing single email during demo
def getIndexedDict():
    with open("indexed100.json", "r") as file:
        return json.load(file)


def load_single_ham_email(id):
    test = pd.read_json("../spam_dataset/test.jsonl", lines=True)
    test_ham = test.loc[test["label"] == 0]
    #print(test_ham)

    return test_ham.loc[test_ham["message_id"] == id]

