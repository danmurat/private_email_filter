from data_functionality.ModelData import ModelData
from data_functionality.ReducedModelData import ReducedModelData
from data_functionality.PreProcess import PreProcess
from ZamaModels import ZamaModels
import util

"""
This file intends to actually train and save any HE compatable ML models,
so that we can use and test (in demo or benchmark.py).
"""

def main():
    # 8th apr testing
    # preprocess_and_save() # working I think.. yes!
    # zamaPlainTrainAndSave() # WORKING! svm=97.55% log=98.15%

    #zamaTrainAndSave() # working
    #zamaTrainAndSaveAndTestWithPca() # working

    # 9th apr: Changes to code-base working. Zama models training and saving as should. No more pre-process repeats.

    print("\n\nModel setup done.\n")


# to quickly load all data later, instead of repeating the pre-process step which takes 15 seconds each time
def preprocess_and_save():
    p = PreProcess()
    p.preprocess()

    print("saving data...")

    X_train, y_train, X_test, y_test = p.getData()
    # model training suddenly stopped liking the y values. Think it's because it's still a pandas datatype?
    y_train = y_train.to_numpy(copy=True)
    y_test = y_test.to_numpy(copy=True)
    model_data = ModelData(X_train, y_train, X_test, y_test)

    red_X_train, red_X_test = util.pca_data(X_train, X_test)
    reduced_model_data = ReducedModelData(red_X_train, red_X_test)

    util.saveModelPickle(model_data, util.model_data_path())
    util.saveModelPickle(reduced_model_data, util.reduced_model_data_path())

    print("data saved! (In pickle_objects/preprocessed_data/)")


def zamaTrainAndSave():
    z = ZamaModels()
    model_data = util.loadModelPickle(util.model_data_path())
    X_train = model_data.get_X_train()
    y_train = model_data.get_y_train()
    X_test = model_data.get_X_test()
    y_test = model_data.get_y_test()

    print("training svm model...")
    svm = z.trainSVM(X_train, y_train)
    z.compileModel(svm, X_train)# run so we can save

    print("training complete!\n")

    print("training logistic regresssion model...")
    log = z.trainLogistic(X_train, y_train)
    z.compileModel(log, X_train)
    print("training complete!\n")

    z.testRandomFHE(svm, X_test, y_test) # seems to work just fine! fhe="execute" not crying
    z.testRandomFHE(log, X_test, y_test)

    print("saving model weights...")
    z.saveModel(svm, "svm")
    print("zama svm saved!")
    z.saveModel(log, "log")
    print("zama logistic regression saved!\n")

def zamaPlainTrainAndSave():
    z = ZamaModels()
    model_data = util.loadModelPickle(util.model_data_path())
    X_train = model_data.get_X_train()
    y_train = model_data.get_y_train()
    #print(y_train)
    X_test = model_data.get_X_test()
    y_test = model_data.get_y_test()

    print("training svm model...")
    svm = z.trainSVM(X_train, y_train)

    print("training complete!\n")

    print("training logistic regresssion model...")
    log = z.trainLogistic(X_train, y_train)
    print("training complete!\n")

    # quick accuracy test to make sure we didn't f up
    print("zama plain svm accuracy: ")
    z.testPlainAccuracy(svm, X_test, y_test)
    print("zama plain logistic accuracy: ")
    z.testPlainAccuracy(log, X_test, y_test)

    print("saving model weights...")
    util.saveModelPickle(svm, "zama_plain_models/svm")
    print("plain zama svm saved!")
    util.saveModelPickle(log, "zama_plain_models/log")
    print("plain zama logistic regression saved!\n")


def zamaTrainAndSaveAndTestWithPca():
    z = ZamaModels()
    model_data = util.loadModelPickle(util.model_data_path())
    reduced_model_data = util.loadModelPickle(util.reduced_model_data_path())
    y_train = model_data.get_y_train()
    red_X_train = reduced_model_data.get_red_X_train()

    print("training pca'd svm model...")
    svm = z.pcaTrainSvm(red_X_train, y_train)
    util.saveModelPickle(svm, "pca_svm") # save plaintext version to quickly test later
    z.pcaCompileModel(svm, red_X_train)

    print("training complete!\n")

    print("training pca'd logistic regresssion model...")
    log = z.pcaTrainLogistic(red_X_train, y_train)
    util.saveModelPickle(log, "pca_log")
    z.pcaCompileModel(log, red_X_train)
    print("training complete!\n")

    print("saving model weights...")
    z.saveModel(svm, "pca_svm")
    print("svm saved!")
    z.saveModel(log, "pca_log")
    print("logistic regression saved!\n")

    pcaLoadPlainZamaAndTest("pca_svm")
    pcaLoadPlainZamaAndTest("pca_log")


def loadPlainZamaModelAndTest(name):
    z = ZamaModels()
    model_data = util.loadModelPickle(util.model_data_path())
    X_test = model_data.get_X_test()
    y_test = model_data.get_y_test()
    print("loading model...")
    model = util.loadModelPickle(name)
    print(f"{name} model loaded!\n")

    print("testing plaintext accuracy...")
    z.testPlainAccuracy(model, X_test, y_test)


def pcaLoadPlainZamaAndTest(name):
    z = ZamaModels()
    model_data = util.loadModelPickle(util.model_data_path())
    reduced_model_data = util.loadModelPickle(util.reduced_model_data_path())
    red_X_test = reduced_model_data.get_red_X_test()
    y_test = model_data.get_y_test()
    print("loading model...")
    model = util.loadModelPickle(name)
    print(f"{name} model loaded!\n")

    print("testing plaintext accuracy...")
    z.pcaTestPlainAccuracy(model, red_X_test, y_test)



if __name__ == "__main__":
    main() 
