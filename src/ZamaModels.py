from typing import Any

from concrete.ml.deployment import FHEModelClient, FHEModelDev, FHEModelServer
from concrete.ml.sklearn import LogisticRegression as ConcreteLogisticRegression
from concrete.ml.sklearn.svm import LinearSVC as ConcreteLinearSVC
from sklearn.metrics import accuracy_score

"""
Handles everything to do with the Zama FHE models (training, predicting, saving, etc..)
"""


class ZamaModels:
    ZAMA_MODELS_PATH = "zama_models/"
    ZAMA_KEYS_PATH = "zama_keys/client_key_"

    # probably delete
    # def pcaReduceEmail(self, email):
    #     pca = self.loadModelPickle("pca")
    #     reduced_email = pca.transform(email)

    #     return reduced_email

    def train_svm(self, X_train, y_train) -> ConcreteLinearSVC:
        svm = ConcreteLinearSVC(max_iter=400, n_bits=8)
        svm.fit(X_train, y_train)

        return svm

    def train_logistic(self, X_train, y_train) -> ConcreteLogisticRegression:
        log = ConcreteLogisticRegression(n_bits=8)
        log.fit(X_train, y_train)

        return log

    # required for fhe
    # this might not work if arguments are passed as copies?
    def compile_model(self, model: Any, X_train) -> None:
        model.compile(X_train)

    def svd_compile_model(self, model: Any, reduced_X_train) -> None:
        model.compile(reduced_X_train)

    def save_model(self, model: Any, name: str) -> None:
        # how do we know what model we're using?? the dir name..
        dev = FHEModelDev(path_dir=self.ZAMA_MODELS_PATH + name, model=model)
        dev.save()

    def load_model(self, name: str) -> FHEModelServer:
        return FHEModelServer(path_dir=self.ZAMA_MODELS_PATH + name)

    def client(self, name) -> tuple:
        client = FHEModelClient(
            path_dir=self.ZAMA_MODELS_PATH + name, key_dir=self.ZAMA_KEYS_PATH + name
        )
        eval_keys = client.get_serialized_evaluation_keys()

        return (client, eval_keys)

    def test_plain_accuracy(self, model: Any, X_test, y_test) -> None:
        model_pred = model.predict(X_test)
        score = accuracy_score(y_test, model_pred)

        print(f"Accuracy = {score}%")

    def pca_test_plain_accuracy(self, model: Any, reduced_X_test, y_test) -> None:
        model_pred = model.predict(reduced_X_test)
        score = accuracy_score(y_test, model_pred)

        print(f"Accuracy = {score}%")

    def classify_plain(self, model: Any, vectorised_text, label) -> None:
        model_pred = model.predict(vectorised_text)
        print(f"Label = {label} | predicted = {model_pred}")

    # quickly checking if fhe on will work outside this class
    def test_random_fhe(self, model: Any, X_test, y_test) -> None:
        model_pred = model.predict(X_test[1100], fhe="execute")

        print(f"label = {y_test[1100]} | predicted = {model_pred}")
