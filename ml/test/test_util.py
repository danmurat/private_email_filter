import numpy as np

import src.utils.constants as c
import src.utils.util as util


def test_svd_data_saves_fitted_transformer(monkeypatch):
    saved = {}

    def save_cloud_model(model, name):
        saved[name] = model

    monkeypatch.setattr(util, "save_cloud_model", save_cloud_model)

    data = np.array(
        [
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0, 4.0],
            [2.0, 0.0, 1.0, 0.0],
        ]
    )
    reduced_train, reduced_test = util.svd_data(data, data[:1], 2)

    assert c.SVD_MODEL_KEY in saved
    assert np.allclose(saved[c.SVD_MODEL_KEY].transform(data), reduced_train)
    assert reduced_test.shape == (1, 2)
