from concrete.ml.sklearn.svm import LinearSVC as ConcreteLinearSVC
from concrete.ml.sklearn import LogisticRegression as ConcreteLogisticRegression
from concrete.ml.deployment import FHEModelDev, FHEModelClient, FHEModelServer
from sklearn.svm import LinearSVC as SklearnLinearSVC
from sklearn.metrics import accuracy_score, f1_score, make_scorer

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

    def trainSVM(self, X_train, y_train) -> ConcreteLinearSVC:
        svm = ConcreteLinearSVC(max_iter=400, n_bits=8)
        svm.fit(X_train, y_train)

        return svm

    def trainLogistic(self, X_train, y_train) -> ConcreteLogisticRegression:
        log = ConcreteLogisticRegression(n_bits=8)
        log.fit(X_train, y_train)

        return log
    

    # required for fhe
    # this might not work if arguments are passed as copies?
    def compileModel(self, model, X_train):
        circuit = model.compile(X_train)
    
    def svdCompileModel(self, model, reduced_X_train):
        circuit = model.compile(reduced_X_train)

    def saveModel(self, model, name):
        # how do we know what model we're using?? the dir name..
        dev = FHEModelDev(path_dir=self.ZAMA_MODELS_PATH + name, model=model)
        dev.save()

    def loadModel(self, name):
        server = FHEModelServer(path_dir=self.ZAMA_MODELS_PATH + name)

        return server
    
    def client(self, name) -> tuple:
        client = FHEModelClient(path_dir=self.ZAMA_MODELS_PATH + name, key_dir=self.ZAMA_KEYS_PATH + name)
        eval_keys = client.get_serialized_evaluation_keys()

        return (client, eval_keys)
        
    def testPlainAccuracy(self, model, X_test, y_test):
        model_pred = model.predict(X_test)
        score = accuracy_score(y_test, model_pred)

        print(f"Accuracy = {score}%")
    
    def pcaTestPlainAccuracy(self, model, reduced_X_test, y_test):
        model_pred = model.predict(reduced_X_test)
        score = accuracy_score(y_test, model_pred)

        print(f"Accuracy = {score}%")


    def classifyPlain(self, model, vectorised_text, label):
        model_pred = model.predict(vectorised_text)
        print(f"Label = {label} | predicted = {model_pred}")

    # quickly checking if fhe on will work outside this class
    def testRandomFHE(self, model, X_test, y_test):
        model_pred = model.predict(X_test[1100], fhe="execute")
        
        print(f"label = {y_test[1100]} | predicted = {model_pred}")