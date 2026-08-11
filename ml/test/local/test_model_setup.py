from pathlib import Path

import pytest

import src.utils.constants as c
from src.execute_scripts import model_setup

"""
Smoke tests for model_setup.py: for every (model, HE scheme) combination we care about
(SVM/LR x zama/TenSEAL/paillier), train + save the model the same way model_setup's
functions already do, then just check the resulting file landed where it's supposed to.

Not testing accuracy/correctness here - just that the train-and-save wiring works.

DO NOT RUN IN CI PIPELINE!!
"""

PICKLE_DIR = Path("models/pickle_objects")
CLOUD_DIR = Path("models/cloud")


@pytest.fixture(scope="module")
def data():
    return model_setup.load_training_data()


@pytest.fixture(scope="module")
def zama_trained(data):
    model_setup.zama_plain_train_and_save(data.X_train, data.y_train, data.X_test, data.y_test)


@pytest.fixture(scope="module")
def tenseal_trained(data):
    model_setup.ts_svd_train_and_save(
        data.t_red_X_train, data.t_y_train, data.t_red_X_test, data.t_y_test
    )


@pytest.fixture(scope="module")
def paillier_trained(tenseal_trained):
    # pal_save() reloads the TenSEAL svd_svm via util.load_model_pickle("ts_plain_models/svd_svm"),
    # but ts_svd_train_and_save() (above) actually writes that model with
    # util.save_cloud_model(..., "svd_svm") -> models/cloud/svd_svm.pkl (cloudpickle, different path
    # and format). So pal_save() only succeeds today because a stale
    # models/pickle_objects/ts_plain_models/svd_svm.pkl already sits on disk from an older run -
    # a latent bug in pal_save(), left as-is here.
    model_setup.pal_save()


def test_zama_svm_saved(zama_trained):
    assert (PICKLE_DIR / c.ZAMA_PLAIN_PATH / f"{c.ZAMA_SVM}.pkl").exists()


def test_zama_lr_saved(zama_trained):
    assert (PICKLE_DIR / c.ZAMA_PLAIN_PATH / f"{c.ZAMA_LOG}.pkl").exists()


def test_tenseal_svm_saved(tenseal_trained):
    assert (CLOUD_DIR / f"{c.TS_SVD_SVM}.pkl").exists()


def test_tenseal_lr_saved(tenseal_trained):
    assert (PICKLE_DIR / c.TS_PLAIN_PATH / f"{c.TS_SVD_LOG}.pkl").exists()


def test_paillier_svm_saved(paillier_trained):
    assert (PICKLE_DIR / c.PAL_PLAIN_PATH / f"{c.PAL_SVD_SVM}.pkl").exists()


def test_paillier_lr_saved(paillier_trained):
    assert (PICKLE_DIR / c.PAL_PLAIN_PATH / f"{c.PAL_SVD_LOG}.pkl").exists()
