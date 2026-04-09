import pickle
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


# pca was saved with n_components=200
def pca_data(X_train, X_test) -> tuple:
    pca = loadModelPickle("pca")
    red_X_train = pca.transform(X_train)
    red_X_test = pca.transform(X_test)

    return (red_X_train, red_X_test)

# incase we ever need to run again/change params
def train_pca():
    pca = PCA(n_components=200)
    saveModelPickle(pca, "pca")


def model_data_path():
    return "preprocessed_data/model_data"

def reduced_model_data_path():
    return "preprocessed_data/reduced_model_data"