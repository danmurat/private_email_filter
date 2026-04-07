from PreProcess import PreProcess
from HandleModel import HandleModel

"""
This file intends to actually train and save any HE compatable ML models,
so that we can use and test (in demo or benchmark.py).
"""

p = PreProcess()
p.preprocess()
h = HandleModel(p)
h.dataSetup()

def trainAndSave():
    print("training svm model...")
    svm = h.trainSVM()
    #h.compileModel(svm) # can't save weights in PICKLE when circuit is compiled!
    # look at gems response on the concrete lib for saving

    print("training complete!\n")

    print("training logistic regresssion model...")
    log = h.trainLogistic()
    #h.compileModel(log)
    print("training complete!\n")

    #h.testAccuracy(svm)

    # of course here we'll actually be saving model weights. But it's worth it to
    # test things first

    #h.testRandomFHE(svm) # seems to work just fine! fhe="execute" not crying

    print("saving model weights...")
    h.saveModelPickle(svm, "svm")
    print("svm saved!")
    h.saveModelPickle(log, "log")
    print("logistic regression saved!\n")
    # print("saving weights as pickle file...")
    # h.saveModelPickle(svm, "svm")


def trainAndSaveAndTestWithPca():
    h.pcaReduce(200)

    print("training pca'd svm model...")
    svm = h.pcaTrainSvm()
    h.saveModelPickle(svm, "pca_svm") # save plaintext version to quickly test later
    h.pcaCompileModel(svm)

    print("training complete!\n")

    print("training pca'd logistic regresssion model...")
    log = h.pcaTrainLogistic()
    h.saveModelPickle(log, "pca_log")
    h.pcaCompileModel(log)
    print("training complete!\n")

    print("saving model weights...")
    h.saveModel(svm, "pca_svm")
    print("svm saved!")
    h.saveModel(log, "pca_log")
    print("logistic regression saved!\n")

    pcaLoadAndTest("pca_svm")
    pcaLoadAndTest("pca_log")




def loadModelAndTest(name):
    print("loading model...")
    model = h.loadModel(name)
    print(f"{name} model loaded!\n")

    print("testing plaintext accuracy...")
    h.testAccuracy(model)


def pcaLoadAndTest(name):
    print("loading model...")
    model = h.loadModelPickle(name)
    print(f"{name} model loaded!\n")

    print("testing plaintext accuracy...")
    h.pcaTestAccuracy(model)



if __name__ == "__main__":
    #trainAndSave()
    #loadModelAndTest()

    # 9th feb PCA testing (above is all done and deployed)
    #trainAndSaveAndTestWithPca()
    #h.pcaReduce(50) # have to recall this since object reduced_X = None otherwise..

    # deploying just plaintext svm/log 
    #trainAndSave()
