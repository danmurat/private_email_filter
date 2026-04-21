from data_functionality.ModelData import ModelData
from data_functionality.ReducedModelData import ReducedModelData
from data_functionality.PreProcess import PreProcess
from ts_compat_models.EncSVM import EncSVM
from ts_compat_models.EncLR import EncLR
from ZamaModels import ZamaModels
from TenSealModels import TenSealModels
from paillier_compat_models.EncLinear import EncLinear
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

    #tsTrainAndSave() # 10th apr test: checking if log working | working!

    #ts_pca_train_and_save() # ts_pca_log = 96.85% | ts_pca_svm = 96%

    #tsTrainAndSave() # working. Models saved under Enc objects
    #ts_pca_train_and_save() # same with pca
    # did we save w and b's properly? YES

    #pal_save() # working as of 4:13pm 11th apr

    # 20th apr (TESTING SVD TOO)
    #preprocess_and_save() # done. svd should be saved correctly
    #zama_plain_test_svd()
    # leaving svd. Likely not compatible with how we've implemented Bag of worms.

    # 21st apr (updated preprocessing to use tfidf alone)
    #preprocess_and_save()
    #zamaTrainAndSave()
    #zamaPlainTrainAndSave()
    tsTrainAndSave()
    #pal_save()
    # p = PreProcess()
    # p.is_data_imbalanced() pretty much 50/50

    # retraining and saving all models with tfidf
    print("\n\nModel setup done.\n")


model_data = util.loadModelPickle(util.model_data_path())
reduced_model_data = util.loadModelPickle(util.reduced_model_data_path())
X_train, y_train, X_test, y_test = model_data.get_all_data()
X_train = X_train.toarray() # sparse can't get passed to models
X_test = X_test.toarray()
red_X_train, red_X_test = reduced_model_data.get_all_data()
t_X_train, t_y_train, t_X_test, t_y_test, t_red_X_train, t_red_X_test = util.convertToTorchTensors(X_train, y_train, X_test, y_test, red_X_train, red_X_test)


# to quickly load all data later, instead of repeating the pre-process step which takes 15 seconds each time
def preprocess_and_save():
    p = PreProcess()
    p.preprocess_tfidf()

    print("saving data...")

    X_train, y_train, X_test, y_test = p.getData()
    # model training suddenly stopped liking the y values. Think it's because it's still a pandas datatype?
    y_train = y_train.to_numpy(copy=True)
    y_test = y_test.to_numpy(copy=True)
    model_data = ModelData(X_train, y_train, X_test, y_test)

    red_X_train, red_X_test = util.svd_data(X_train, X_test, 200)

    reduced_model_data = ReducedModelData(red_X_train, red_X_test)

    util.saveModelPickle(model_data, util.model_data_path())
    util.saveModelPickle(reduced_model_data, util.reduced_model_data_path())

    print("data saved! (In pickle_objects/preprocessed_data/)")


def zamaTrainAndSave():
    z = ZamaModels()

    print("training svm model...")
    svm = z.trainSVM(X_train, y_train)
    z.compileModel(svm, X_train)# run so we can save

    print("training complete!\n")

    print("training logistic regression model...")
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


def zama_plain_test_svd():
    z = ZamaModels()

    print("training svm model...")
    svm = z.trainSVM(red_X_train, y_train)

    print("training complete!\n")

    print("training logistic regresssion model...")
    log = z.trainLogistic(red_X_train, y_train)
    print("training complete!\n")

    # quick accuracy test to make sure we didn't f up
    print("zama plain svm accuracy: ")
    z.testPlainAccuracy(svm, red_X_test, y_test)
    print("zama plain logistic accuracy: ")
    z.testPlainAccuracy(log, red_X_test, y_test)


def zamaTrainAndSaveAndTestWithSvd():
    z = ZamaModels()

    print("training pca'd svm model...")
    svm = z.trainSVM(red_X_train, y_train)
    util.saveModelPickle(svm, "svd_svm") # save plaintext version to quickly test later
    z.pcaCompileModel(svm, red_X_train)

    print("training complete!\n")

    print("training pca'd logistic regresssion model...")
    log = z.trainLogistic(red_X_train, y_train)
    util.saveModelPickle(log, "svd_log")
    z.pcaCompileModel(log, red_X_train)
    print("training complete!\n")

    print("saving model weights...")
    z.saveModel(svm, "svd_svm")
    print("svm saved!")
    z.saveModel(log, "svd_log")
    print("logistic regression saved!\n")

    svdLoadPlainZamaAndTest("svd_svm")
    svdLoadPlainZamaAndTest("svd_log")


def loadPlainZamaModelAndTest(name):
    z = ZamaModels()

    print("loading model...")
    model = util.loadModelPickle(name)
    print(f"{name} model loaded!\n")

    print("testing plaintext accuracy...")
    z.testPlainAccuracy(model, X_test, y_test)


def svdLoadPlainZamaAndTest(name):
    z = ZamaModels()

    print("loading model...")
    model = util.loadModelPickle(name)
    print(f"{name} model loaded!\n")

    print("testing plaintext accuracy...")
    z.pcaTestPlainAccuracy(model, red_X_test, y_test)


# now tenseal compat models

def tsTrainAndSave():
    ts = TenSealModels()

    # commenting out log to deal with low svm accuracy on tfidf
    print("Training tenseal logistic regression...")
    ts_pre_log = ts.trainLog(t_X_train, t_y_train, 3000) # 97.55% acc
    ts_log = EncLR(ts_pre_log) # here is where we save the weights and allow for encrypted inference
    print("training finished.")

    ts.torch_log_predictions(ts_pre_log, t_X_test, t_y_test) # prints accuracy (throw y_pred away)

    # print("Training tenseal svm...")
    # ts_pre_svm = ts.trainSVM(t_X_train, t_y_train, 2300) # 97%
    # ts_svm = EncSVM(ts_pre_svm)
    # print("training finished.")
    #
    # ts.svmAccuracy(ts_pre_svm, t_X_test, t_y_test)

    print("Saving models with pickle...")
    util.saveModelPickle(ts_log, "ts_plain_models/log")
    #util.saveModelPickle(ts_svm, "ts_plain_models/svm")
    print("Tenseal models saved.")

def ts_svd_train_and_save():
    ts = TenSealModels()
    print("Training tenseal logistic regression...")
    ts_pre_svd_log = ts.trainLog(t_red_X_train, t_y_train, 3000) # 97.55% acc
    ts_svd_log = EncLR(ts_pre_svd_log) # here is where we save the weights and allow for encrypted inference
    print("training finished.")

    ts.logAccuracy(ts_pre_svd_log, t_red_X_test, t_y_test)

    print("Training tenseal svm...")
    ts_pre_svd_svm = ts.trainSVM(t_red_X_train, t_y_train, 5000) # 97%
    ts_svd_svm = EncSVM(ts_pre_svd_svm)
    print("training finished.")

    ts.svmAccuracy(ts_pre_svd_svm, t_red_X_test, t_y_test)

    print("Saving models with pickle...")
    util.saveModelPickle(ts_svd_log, "ts_plain_models/svd_log")
    util.saveModelPickle(ts_svd_svm, "ts_plain_models/svd_svm")
    print("Tenseal models saved.")

# can just load ts_plain_models (already trained) into EncLinear. All we need is weights/bias
def pal_save():
    print("Loading ts compat models for paillier...")
    ts_log = util.loadModelPickle("ts_plain_models/log")
    #ts_svd_log = util.loadModelPickle("ts_plain_models/svd_log")
    ts_svm = util.loadModelPickle("ts_plain_models/svm")
    #ts_svd_svm = util.loadModelPickle("ts_plain_models/svd_svm")
    print("Loaded.")

    # confirm that w's are appropriate lengths
    print(len(ts_log.w))
    #print(len(ts_svd_log.w))
    print(len(ts_svm.w))
    #print(len(ts_svd_svm.w))

    pal_log = EncLinear(ts_log)
    #pal_pca_log = EncLinear(ts_svd_log)
    pal_svm = EncLinear(ts_svm)
    #pal_pca_svm = EncLinear(ts_svd_svm)

    print("Saving paillier models...")
    util.saveModelPickle(pal_log, "pal_plain_models/log")
    #util.saveModelPickle(pal_pca_log, "pal_plain_models/svd_log")
    util.saveModelPickle(pal_svm, "pal_plain_models/svm")
    #util.saveModelPickle(pal_pca_svm, "pal_plain_models/svd_svm")
    print("Models saved.")


# added EncSVM to detach.numpy on parameter selection. So Paillier should automatically work for both ts_log and svm
def resave_ts_svm():
    ts_pre_svm = util.loadModelPickle("ts_plain_models/svm")
    ts_pre_svd_svm = util.loadModelPickle("ts_plain_models/svd_svm")

    print(type(ts_pre_svd_svm.w))
    print(ts_pre_svd_svm.w)
    print(type(ts_pre_svd_svm.b))
    #
    # ts_svm = EncSVM(ts_pre_svm)
    # ts_pca_svm = EncSVM(ts_pre_pca_svm)
    #
    # # now saved objects can call the EncSVM object methods
    # util.saveModelPickle(ts_svm, "ts_plain_models/svm")
    # util.saveModelPickle(ts_pca_svm, "ts_plain_models/pca_svm")

if __name__ == "__main__":
    main() 
