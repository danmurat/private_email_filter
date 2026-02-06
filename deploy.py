from PreProcess import PreProcess
from HandleModel import HandleModel

p = PreProcess()
p.preprocess()
h = HandleModel(p)
h.dataSetup()

def trainAndSave():
    print("training svm model...")
    svm = h.trainSVM()
    h.compileModel(svm) # can't save weights in PICKLE when circuit is compiled!
    # look at gems response on the concrete lib for saving

    print("training complete!\n")

    #h.testAccuracy(svm)

    # of course here we'll actually be saving model weights. But it's worth it to
    # test things first

    #h.testRandomFHE(svm) # seems to work just fine! fhe="execute" not crying

    print("saving model weights...")
    h.saveModel(svm, "svm")
    # print("saving weights as pickle file...")
    # h.saveModelPickle(svm, "svm")

    

    print("model weights saved!\n")


def loadModelAndTest():
    print("loading model...")
    svm = h.loadModel("svm")
    print("model loaded!\n")

    print("testing plaintext accuracy...")
    h.testAccuracy(svm)

    # print("\ntesting encrypted inference...")
    # h.testRandomFHE(svm)  

if __name__ == "__main__":
    trainAndSave()
    #loadModelAndTest()

