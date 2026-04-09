import pickle
import torch
import pandas as pd
import json
import tenseal as ts
from sklearn.decomposition import PCA

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
def pca_data(X_train, X_test) -> tuple:
    pca = loadModelPickle("pca")
    red_X_train = pca.transform(X_train)
    red_X_test = pca.transform(X_test)

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


"""
IMPORTANT

Making sure these params are correct allows the actual encrypted operations to work. Reason why we were
wildly off was because our input wasn't fitting in the cyphertext, so was missing a lot of info!

The numbers given below also determine how efficient and secure the scheme runs. How does it work? I don't know.
I guess I'll play around with the numbers at first, until I can find some known optimisations to place.

PCA'd data may be interesting here? for it to run VERY fast?
"""
def setup_ts_params_svm():
    # from tenseal docs:
    # parameters
    poly_mod_degree = 2 ** 14 # 2^12 = 4096 basic (must be pow 2) -- lower = more efficient
    coeff_mod_bit_sizes = [40, 40, 40, 40]
    # create TenSEALContext
    ctx_eval = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
    # scale of ciphertext to use
    ctx_eval.global_scale = 2 ** 40
    # this key is needed for doing dot-product operations
    ctx_eval.generate_galois_keys()

    return ctx_eval

