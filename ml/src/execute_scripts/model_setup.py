import src.utils.constants as c
import src.utils.util as util
import cloudpickle
from src.data_functionality.ModelData import ModelData
from src.data_functionality.ReducedModelData import ReducedModelData
from src.data_functionality.PreProcess import PreProcess
# so we can save these modules for cloudpickle
import src.data_functionality.ModelData
import src.data_functionality.ReducedModelData
from src.paillier_compat_models.EncLinear import EncLinear
from src.TenSealModels import TenSealModels
from src.ts_compat_models.EncLR import EncLR
from src.ts_compat_models.EncSVM import EncSVM
import src.ts_compat_models.EncSVM # same thing applies here (as model data's)
from src.ZamaModels import ZamaModels

"""
This file intends to actually train and save any HE compatable ML models,
so that we can use and test (in demo or benchmark.py).

TODO:
this file is a bit gross. We have many functions training, saving or loading -
these can be much more 'injected'. So we can inject the model we want to train, inject
the data (complete, or reduced), etc.
This could do with a big cleanup.
"""


def main() -> None:
    # 8th apr testing
    # preprocess_and_save() # working I think.. yes!
    # zamaPlainTrainAndSave() # WORKING! svm=97.55% log=98.15%

    # zamaTrainAndSave() # working
    # zamaTrainAndSaveAndTestWithPca() # working

    # 9th apr: Changes to code-base working. Zama models training and saving as should. No more pre-process repeats.

    # tsTrainAndSave() # 10th apr test: checking if log working | working!

    # ts_pca_train_and_save() # ts_pca_log = 96.85% | ts_pca_svm = 96%

    # tsTrainAndSave() # working. Models saved under Enc objects
    # ts_pca_train_and_save() # same with pca
    # did we save w and b's properly? YES

    # pal_save() # working as of 4:13pm 11th apr

    # 20th apr (TESTING SVD TOO)
    # preprocess_and_save() # done. svd should be saved correctly
    # zama_plain_test_svd()
    # leaving svd. Likely not compatible with how we've implemented Bag of worms.

    # 21st apr (updated preprocessing to use tfidf alone)
    # preprocess_and_save()
    # zamaTrainAndSave()
    # zamaPlainTrainAndSave()
    # tsTrainAndSave()
    # pal_save()
    # p = PreProcess()
    # p.is_data_imbalanced() pretty much 50/50

    # 22nd apr (need to set up models on reduced data for testing)
    # preprocess_and_save() # run again to make sure we're using svd with 200 components
    # zamaTrainAndSaveAndTestWithSvd()
    # ts_svd_train_and_save()
    # pal_save()

    # re-running after restructuring project directories (pickle didn't like it).
    #preprocess_and_save()
    #ts_svd_train_and_save() # THIS IS THE ONLY MODEL THAT ACTUALLY LOADS AS OF 14/06/2026
    print("\n\nModel setup done.\n")


model_data = util.load_model_pickle(c.MODEL_DATA_PATH)
reduced_model_data = util.load_model_pickle(c.REDUCED_MODEL_DATA_PATH)
X_train, y_train, X_test, y_test = model_data.get_all_data()
X_train = X_train.toarray()  # sparse can't get passed to models
X_test = X_test.toarray()
red_X_train, red_X_test = reduced_model_data.get_all_data()
t_X_train, t_y_train, t_X_test, t_y_test, t_red_X_train, t_red_X_test = (
    util.convert_to_torch_tensors(
        X_train, y_train, X_test, y_test, red_X_train, red_X_test
    )
)


# to quickly load all data later, instead of repeating the pre-process step which takes 15 seconds each time
def preprocess_and_save() -> None:
    p = PreProcess()
    p.preprocess_tfidf()

    print("saving data...")

    X_train, y_train, X_test, y_test = p.get_data()
    # model training suddenly stopped liking the y values. Think it's because it's still a pandas datatype?
    y_train = y_train.to_numpy(copy=True)
    y_test = y_test.to_numpy(copy=True)
    model_data = ModelData(X_train, y_train, X_test, y_test)

    red_X_train, red_X_test = util.svd_data(X_train, X_test, 150)

    reduced_model_data = ReducedModelData(red_X_train, red_X_test)

    cloudpickle.register_pickle_by_value(src.data_functionality.ModelData)
    cloudpickle.register_pickle_by_value(src.data_functionality.ReducedModelData)
    util.save_cloud_model(model_data, c.MODEL_DATA_KEY)
    util.save_cloud_model(reduced_model_data, c.REDUCED_MODEL_DATA_KEY)

    #print("data saved! (In pickle_objects/preprocessed_data/)")
    print("data saved! (In models/cloud/)")


def zama_train_and_save() -> None:
    z = ZamaModels()

    print("training svm model...")
    svm = z.train_svm(X_train, y_train)
    z.compile_model(svm, X_train)  # run so we can save

    print("training complete!\n")

    print("training logistic regression model...")
    log = z.train_logistic(X_train, y_train)
    z.compile_model(log, X_train)
    print("training complete!\n")

    z.test_random_fhe(
        svm, X_test, y_test
    )  # seems to work just fine! fhe="execute" not crying
    z.test_random_fhe(log, X_test, y_test)

    print("saving model weights...")
    z.save_model(svm, "svm")
    print("zama svm saved!")
    z.save_model(log, "log")
    print("zama logistic regression saved!\n")


def zama_plain_train_and_save() -> None:
    z = ZamaModels()

    print("training svm model...")
    svm = z.train_svm(X_train, y_train)

    print("training complete!\n")

    print("training logistic regresssion model...")
    log = z.train_logistic(X_train, y_train)
    print("training complete!\n")

    # quick accuracy test to make sure we didn't f up
    print("zama plain svm accuracy: ")
    z.test_plain_accuracy(svm, X_test, y_test)
    print("zama plain logistic accuracy: ")
    z.test_plain_accuracy(log, X_test, y_test)

    print("saving model weights...")
    util.save_model_pickle(svm, c.ZAMA_PLAIN_PATH + c.ZAMA_SVM)
    print("plain zama svm saved!")
    util.save_model_pickle(log, c.ZAMA_PLAIN_PATH + c.ZAMA_LOG)
    print("plain zama logistic regression saved!\n")


def zama_plain_test_svd() -> None:
    z = ZamaModels()

    print("training svm model...")
    svm = z.train_svm(red_X_train, y_train)

    print("training complete!\n")

    print("training logistic regresssion model...")
    log = z.train_logistic(red_X_train, y_train)
    print("training complete!\n")

    # quick accuracy test to make sure we didn't f up
    print("zama plain svm accuracy: ")
    z.test_plain_accuracy(svm, red_X_test, y_test)
    print("zama plain logistic accuracy: ")
    z.test_plain_accuracy(log, red_X_test, y_test)


def zama_train_save_test_with_svd() -> None:
    z = ZamaModels()

    print("training pca'd svm model...")
    svd_svm = z.train_svm(red_X_train, y_train)
    util.save_model_pickle(
        svd_svm, c.ZAMA_PLAIN_PATH + c.ZAMA_SVD_SVM
    )  # save plaintext version to quickly test later
    z.svd_compile_model(svd_svm, red_X_train)

    print("training complete!\n")

    print("training pca'd logistic regresssion model...")
    svd_log = z.train_logistic(red_X_train, y_train)
    util.save_model_pickle(svd_log, c.ZAMA_PLAIN_PATH + c.ZAMA_SVD_LOG)
    z.svd_compile_model(svd_log, red_X_train)
    print("training complete!\n")

    print("saving model weights...")
    z.save_model(svd_svm, c.ZAMA_SVD_SVM)
    print("svm saved!")
    z.save_model(svd_log, c.ZAMA_SVD_LOG)
    print("logistic regression saved!\n")

    svd_load_plain_zama_and_test(c.ZAMA_PLAIN_PATH, c.ZAMA_SVD_SVM)
    svd_load_plain_zama_and_test(c.ZAMA_PLAIN_PATH, c.ZAMA_SVD_LOG)


def load_plain_zama_model_and_test(model_path: str, model_name: str) -> None:
    z = ZamaModels()

    print("loading model...")
    model = util.load_model_pickle(model_path + model_name)
    print(f"{model_name} model loaded!\n")

    print("testing plaintext accuracy...")
    z.test_plain_accuracy(model, X_test, y_test)


def svd_load_plain_zama_and_test(model_path: str, model_name: str) -> None:
    z = ZamaModels()

    print("loading model...")
    model = util.load_model_pickle(model_path + model_name)
    print(f"{model_name} model loaded!\n")

    print("testing plaintext accuracy...")
    z.pca_test_plain_accuracy(model, red_X_test, y_test)


# now tenseal compat models


def ts_train_and_save() -> None:
    ts = TenSealModels()

    # commenting out log to deal with low svm accuracy on tfidf
    print("Training tenseal logistic regression...")
    ts_pre_log = ts.train_log(t_X_train, t_y_train, 3000)  # 97.55% acc
    ts_log = EncLR(
        ts_pre_log
    )  # here is where we save the weights and allow for encrypted inference
    print("training finished.")

    ts.torch_log_predictions(
        ts_pre_log, t_X_test, t_y_test
    )  # prints accuracy (throw y_pred away)

    # print("Training tenseal svm...")
    # ts_pre_svm = ts.trainSVM(t_X_train, t_y_train, 2300) # 97%
    # ts_svm = EncSVM(ts_pre_svm)
    # print("training finished.")
    #
    # ts.svmAccuracy(ts_pre_svm, t_X_test, t_y_test)

    print("Saving models with pickle...")
    util.save_model_pickle(ts_log, c.TS_PLAIN_PATH + c.TS_LOG)
    # util.saveModelPickle(ts_svm, "ts_plain_models/svm")
    print("Tenseal models saved.")


def ts_svd_train_and_save() -> None:
    ts = TenSealModels()
    print("Training tenseal logistic regression...")
    ts_pre_svd_log = ts.train_log(t_red_X_train, t_y_train, 3000)  # 97.55% acc
    ts_svd_log = EncLR(
        ts_pre_svd_log
    )  # here is where we save the weights and allow for encrypted inference
    print("training finished.")

    ts.torch_log_predictions(ts_pre_svd_log, t_red_X_test, t_y_test)

    print("Training tenseal svm...")
    ts_pre_svd_svm = ts.train_svm(t_red_X_train, t_y_train, 2200)  # 97%
    ts_svd_svm = EncSVM(ts_pre_svd_svm)
    print("training finished.")

    ts.svm_predictions(ts_pre_svd_svm, t_red_X_test, t_y_test)

    print("Saving models with pickle...")
    util.save_model_pickle(ts_svd_log, c.TS_PLAIN_PATH + c.TS_SVD_LOG)
    # saving svd log as cloud model!
    cloudpickle.register_pickle_by_value(src.ts_compat_models.EncSVM)
    util.save_cloud_model(ts_svd_svm, c.TS_SVD_SVM)
    print("Tenseal models saved.")


# can just load ts_plain_models (already trained) into EncLinear. All we need is weights/bias
def pal_save() -> None:
    # SAVING JUST REDUCED DATA VERSIONS
    print("Loading ts compat models for paillier...")
    # ts_log = util.loadModelPickle("ts_plain_models/log")
    ts_svd_log = util.load_model_pickle(c.TS_PLAIN_PATH + c.TS_SVD_LOG)
    # ts_svm = util.loadModelPickle("ts_plain_models/svm")
    ts_svd_svm = util.load_model_pickle(c.TS_PLAIN_PATH + c.TS_SVD_SVM)
    print("Loaded.")

    # confirm that w's are appropriate lengths
    # print(len(ts_log.w))
    print(len(ts_svd_log.w))
    # print(len(ts_svm.w))
    print(len(ts_svd_svm.w))

    # pal_log = EncLinear(ts_log)
    pal_svd_log = EncLinear(ts_svd_log)
    # pal_svm = EncLinear(ts_svm)
    pal_svd_svm = EncLinear(ts_svd_svm)

    print("Saving paillier models...")
    # util.saveModelPickle(pal_log, "pal_plain_models/log")
    util.save_model_pickle(pal_svd_log, c.PAL_PLAIN_PATH + c.PAL_SVD_LOG)
    # util.saveModelPickle(pal_svm, "pal_plain_models/svm")
    util.save_model_pickle(pal_svd_svm, c.PAL_PLAIN_PATH + c.PAL_SVD_SVM)
    print("Models saved.")


# added EncSVM to detach.numpy on parameter selection. So Paillier should automatically work for both ts_log and svm
def resave_ts_svm() -> None:
    ts_pre_svm = util.load_model_pickle(c.TS_PLAIN_PATH + c.TS_SVM)
    ts_pre_svd_svm = util.load_model_pickle(c.TS_PLAIN_PATH + c.TS_SVD_SVM)

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
