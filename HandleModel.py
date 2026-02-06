from concrete.ml.sklearn.svm import LinearSVC as ConcreteLinearSVC
from concrete.ml.sklearn import LogisticRegression as ConcreteLogisticRegression
from concrete.ml.deployment import FHEModelDev, FHEModelClient, FHEModelServer
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from PreProcess import PreProcess
import pickle

# utility'esque class that handles training, saving, testing + other stuff
class HandleModel:
    # p is our preprocessed object
    def __init__(self, p: PreProcess):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.p = p

        #self.p.preprocess() # run outside method!
        #self.dataSetup() # instantiates above variables. Run outside too!

    def dataSetup(self):
        self.X_train = self.p.getVectorisedTrainText()
        self.y_train = self.p.getTrainingData()["label"]
        self.X_test = self.p.getVectorisedTestText()
        self.y_test = self.p.getTestingData()["label"]


    def trainSVM(self) -> ConcreteLinearSVC:
        svm = ConcreteLinearSVC(max_iter=400, n_bits=8)
        svm.fit(self.X_train, self.y_train)

        return svm

    def trainLogistic(self) -> ConcreteLogisticRegression:
        log = ConcreteLogisticRegression(n_bits=8)
        log.fit(self.X_train, self.y_train)

        return log

    # required for fhe
    # this might not work if arguments are passed as copies?
    def compileModel(self, model):
        circuit = model.compile(self.X_train)

    def saveModelPickle(self, model, name):
        with open(f"{name}.pkl", "wb") as file:
            pickle.dump(model, file)

    def loadModelPickle(self, name):
        with open(f"{name}.pkl", "rb") as file:
            return pickle.load(file)

    def saveModel(self, model, name):
        # how do we know what model we're using?? the dir name..
        dev = FHEModelDev(path_dir=name, model=model)
        dev.save()

    def loadModel(self, name):
        server = FHEModelServer(path_dir=name)

        return server
    
    def testAccuracy(self, model):
        model_pred = model.predict(self.X_test)
        score = accuracy_score(self.y_test, model_pred)

        print(f"Accuracy = {score}%")

    def classifyPlain(self, model, vectorised_text, label):
        model_pred = model.predict(vectorised_text)
        print(f"Label = {label} | predicted = {model_pred}")

    # quickly checking if fhe on will work outside this class
    def testRandomFHE(self, model):
        model_pred = model.predict(self.X_test[1100], fhe="execute")
        
        print(f"label = {self.y_test[1100]} | predicted = {model_pred}")



    # then potentially have inference testing here? Not sure if this being a class
    # can bottleneck things?