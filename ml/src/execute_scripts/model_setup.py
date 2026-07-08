from dataclasses import dataclass
from typing import Any

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
Trains and saves any HE compatible ML models, so that we can use and test them
(in demo or benchmark.py).

Nothing here reads from module-level globals: data is loaded explicitly via
load_training_data() and passed into whichever train/save function needs it.
"""


@dataclass
class TrainingData:
    X_train: Any
    y_train: Any
    X_test: Any
    y_test: Any
    red_X_train: Any
    red_X_test: Any
    t_X_train: Any
    t_y_train: Any
    t_X_test: Any
    t_y_test: Any
    t_red_X_train: Any
    t_red_X_test: Any


def load_training_data() -> TrainingData:
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

    return TrainingData(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        red_X_train=red_X_train,
        red_X_test=red_X_test,
        t_X_train=t_X_train,
        t_y_train=t_y_train,
        t_X_test=t_X_test,
        t_y_test=t_y_test,
        t_red_X_train=t_red_X_train,
        t_red_X_test=t_red_X_test,
    )


def main() -> None:
    preprocess_and_save()

    data = load_training_data()

    zama_train_and_save(data.X_train, data.y_train, data.X_test, data.y_test)
    zama_plain_train_and_save(data.X_train, data.y_train, data.X_test, data.y_test)
    zama_train_save_test_with_svd(
        data.red_X_train, data.y_train, data.red_X_test, data.y_test
    )

    ts_train_and_save(data.t_X_train, data.t_y_train, data.t_X_test, data.t_y_test)
    ts_svd_train_and_save(
        data.t_red_X_train, data.t_y_train, data.t_red_X_test, data.t_y_test
    )

    pal_save()

    print("\n\nModel setup done.\n")


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

    print("data saved! (In models/cloud/)")


# ---- zama (FHE-compiled) models ----


def _train_compile_save_fhe_zama(
    z: ZamaModels, train_fn, compile_fn, model_name: str, X_train, y_train, X_test, y_test
) -> Any:
    print(f"training {model_name} model...")
    model = train_fn(X_train, y_train)
    compile_fn(model, X_train)  # run so we can save
    print("training complete!\n")

    z.test_random_fhe(model, X_test, y_test)

    print("saving model weights...")
    z.save_model(model, model_name)
    print(f"zama {model_name} saved!\n")

    return model


def zama_train_and_save(X_train, y_train, X_test, y_test) -> None:
    z = ZamaModels()

    _train_compile_save_fhe_zama(
        z, z.train_svm, z.compile_model, c.ZAMA_SVM, X_train, y_train, X_test, y_test
    )
    _train_compile_save_fhe_zama(
        z, z.train_logistic, z.compile_model, c.ZAMA_LOG, X_train, y_train, X_test, y_test
    )


def _train_save_fhe_zama_svd(z: ZamaModels, train_fn, model_name: str, red_X_train, y_train) -> Any:
    print(f"training pca'd {model_name} model...")
    model = train_fn(red_X_train, y_train)
    util.save_model_pickle(
        model, c.ZAMA_PLAIN_PATH + model_name
    )  # save plaintext version to quickly test later
    z.svd_compile_model(model, red_X_train)
    print("training complete!\n")

    print("saving model weights...")
    z.save_model(model, model_name)
    print(f"{model_name} saved!\n")

    return model


def zama_train_save_test_with_svd(red_X_train, y_train, red_X_test, y_test) -> None:
    z = ZamaModels()

    _train_save_fhe_zama_svd(z, z.train_svm, c.ZAMA_SVD_SVM, red_X_train, y_train)
    _train_save_fhe_zama_svd(z, z.train_logistic, c.ZAMA_SVD_LOG, red_X_train, y_train)

    svd_load_plain_zama_and_test(c.ZAMA_PLAIN_PATH, c.ZAMA_SVD_SVM, red_X_test, y_test)
    svd_load_plain_zama_and_test(c.ZAMA_PLAIN_PATH, c.ZAMA_SVD_LOG, red_X_test, y_test)


# ---- zama (plaintext) models ----


def _train_and_test_plain_zama(
    z: ZamaModels, train_fn, model_name: str, X_train, y_train, X_test, y_test
) -> Any:
    print(f"training {model_name} model...")
    model = train_fn(X_train, y_train)
    print("training complete!\n")

    print(f"zama plain {model_name} accuracy: ")
    z.test_plain_accuracy(model, X_test, y_test)

    return model


def _train_and_save_plain_zama(
    z: ZamaModels, train_fn, model_name: str, X_train, y_train, X_test, y_test
) -> Any:
    model = _train_and_test_plain_zama(z, train_fn, model_name, X_train, y_train, X_test, y_test)

    print("saving model weights...")
    util.save_model_pickle(model, c.ZAMA_PLAIN_PATH + model_name)
    print(f"plain zama {model_name} saved!\n")

    return model


def zama_plain_train_and_save(X_train, y_train, X_test, y_test) -> None:
    z = ZamaModels()

    _train_and_save_plain_zama(z, z.train_svm, c.ZAMA_SVM, X_train, y_train, X_test, y_test)
    _train_and_save_plain_zama(z, z.train_logistic, c.ZAMA_LOG, X_train, y_train, X_test, y_test)


def zama_plain_test_svd(red_X_train, y_train, red_X_test, y_test) -> None:
    z = ZamaModels()

    _train_and_test_plain_zama(z, z.train_svm, c.ZAMA_SVM, red_X_train, y_train, red_X_test, y_test)
    _train_and_test_plain_zama(
        z, z.train_logistic, c.ZAMA_LOG, red_X_train, y_train, red_X_test, y_test
    )


def load_plain_zama_model_and_test(model_path: str, model_name: str, X_test, y_test) -> None:
    z = ZamaModels()

    print("loading model...")
    model = util.load_model_pickle(model_path + model_name)
    print(f"{model_name} model loaded!\n")

    print("testing plaintext accuracy...")
    z.test_plain_accuracy(model, X_test, y_test)


def svd_load_plain_zama_and_test(model_path: str, model_name: str, red_X_test, y_test) -> None:
    z = ZamaModels()

    print("loading model...")
    model = util.load_model_pickle(model_path + model_name)
    print(f"{model_name} model loaded!\n")

    print("testing plaintext accuracy...")
    z.pca_test_plain_accuracy(model, red_X_test, y_test)


# ---- tenseal compat models ----


def _train_test_ts_log(ts: TenSealModels, t_X_train, t_y_train, t_X_test, t_y_test, epochs=3000) -> EncLR:
    print("Training tenseal logistic regression...")
    ts_pre_log = ts.train_log(t_X_train, t_y_train, epochs)  # 97.55% acc
    ts_log = EncLR(ts_pre_log)  # here is where we save the weights and allow for encrypted inference
    print("training finished.")

    ts.torch_log_predictions(ts_pre_log, t_X_test, t_y_test)  # prints accuracy (throw y_pred away)

    return ts_log


def ts_train_and_save(t_X_train, t_y_train, t_X_test, t_y_test) -> None:
    ts = TenSealModels()

    ts_log = _train_test_ts_log(ts, t_X_train, t_y_train, t_X_test, t_y_test)

    print("Saving models with pickle...")
    util.save_model_pickle(ts_log, c.TS_PLAIN_PATH + c.TS_LOG)
    print("Tenseal models saved.")


def ts_svd_train_and_save(t_red_X_train, t_y_train, t_red_X_test, t_y_test) -> None:
    ts = TenSealModels()

    ts_svd_log = _train_test_ts_log(ts, t_red_X_train, t_y_train, t_red_X_test, t_y_test)

    print("Training tenseal svm...")
    ts_pre_svd_svm = ts.train_svm(t_red_X_train, t_y_train, 2200)  # 97%
    ts_svd_svm = EncSVM(ts_pre_svd_svm)
    print("training finished.")

    ts.svm_predictions(ts_pre_svd_svm, t_red_X_test, t_y_test)

    print("Saving models with pickle...")
    util.save_model_pickle(ts_svd_log, c.TS_PLAIN_PATH + c.TS_SVD_LOG)
    util.save_model_pickle(ts_svd_log, c.TS_PLAIN_PATH + c.TS_SVD_SVM)
    # also saving svd log as cloud model! (for server)
    cloudpickle.register_pickle_by_value(src.ts_compat_models.EncSVM)
    util.save_cloud_model(ts_svd_svm, c.TS_SVD_SVM)
    print("Tenseal models saved.")


# can just load ts_plain_models (already trained) into EncLinear. All we need is weights/bias
def pal_save() -> None:
    # SAVING JUST REDUCED DATA VERSIONS
    print("Loading ts compat models for paillier...")
    ts_svd_log = util.load_model_pickle(c.TS_PLAIN_PATH + c.TS_SVD_LOG)
    ts_svd_svm = util.load_model_pickle(c.TS_PLAIN_PATH + c.TS_SVD_SVM)
    print("Loaded.")

    # confirm that w's are appropriate lengths
    print(len(ts_svd_log.w))
    print(len(ts_svd_svm.w))

    pal_svd_log = EncLinear(ts_svd_log)
    pal_svd_svm = EncLinear(ts_svd_svm)

    print("Saving paillier models...")
    util.save_model_pickle(pal_svd_log, c.PAL_PLAIN_PATH + c.PAL_SVD_LOG)
    util.save_model_pickle(pal_svd_svm, c.PAL_PLAIN_PATH + c.PAL_SVD_SVM)
    print("Models saved.")


# added EncSVM to detach.numpy on parameter selection. So Paillier should automatically work for both ts_log and svm
def resave_ts_svm() -> None:
    ts_pre_svm = util.load_model_pickle(c.TS_PLAIN_PATH + c.TS_SVM)
    ts_pre_svd_svm = util.load_model_pickle(c.TS_PLAIN_PATH + c.TS_SVD_SVM)

    print(type(ts_pre_svd_svm.w))
    print(ts_pre_svd_svm.w)
    print(type(ts_pre_svd_svm.b))


if __name__ == "__main__":
    main()
