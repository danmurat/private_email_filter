import numpy as np
from sklearn.metrics import classification_report

import src.utils.util as util
from src.TenSealModels import TenSealModels
import src.utils.constants as c

# DATA
model_data = util.load_model_pickle(c.MODEL_DATA_PATH)
reduced_model_data = util.load_model_pickle(c.REDUCED_MODEL_DATA_PATH)

X_test = model_data.get_X_test()
y_test = model_data.get_y_test()
X_test = X_test.toarray()  # remember these are sparse when loaded
red_X_test = reduced_model_data.get_red_X_test()

t_X_train, t_y_train, t_X_test, t_y_test, t_red_X_train, t_red_X_test = (
    util.convert_to_torch_tensors(
        np.zeros((0, 0)), np.zeros((0, 0)), X_test, y_test, np.zeros((0, 0)), red_X_test
    )
)

"""
Seperately evaluating predictive performance of all plaintext models.
(encrypted prediction is equivalent, so no need to waste time).
"""


def main() -> None:
    zama_report(c.ZAMA_PLAIN_PATH, c.ZAMA_SVD_LOG)
    zama_report(c.ZAMA_PLAIN_PATH, c.ZAMA_SVD_SVM)
    ts_report(c.TS_PLAIN_PATH, c.TS_SVD_LOG)
    ts_report(c.TS_PLAIN_PATH, c.TS_SVD_SVM)


def zama_report(model_path: str, model_name: str) -> None:
    model = util.load_model_pickle(model_path + model_name)
    y_pred = model.predict(red_X_test) if _is_svd(model_name) else model.predict(X_test)

    print(f"\nZAMA {model_name} REPORT:\n")
    print(classification_report(y_test, y_pred, digits=4))


def ts_report(model_path: str, model_name: str) -> None:
    t = TenSealModels()  # to run the accuracy tests
    model = util.load_model_pickle(model_path + model_name)

    is_log = model_name[0] == "l" or (
        len(model_name) > 3 and model_name[4] == "l"
    )  # not sure if [5] doesn't exist, will break?

    t_x = t_red_X_test if _is_svd(model_name) else t_X_test
    x = red_X_test if _is_svd(model_name) else X_test

    y_pred = None
    if is_log:
        y_pred = t.log_predictions(model, t_x)
        # lr saves results as floats after the sigmoid, so need to convert to 0 or 1
        y_pred = [0 if x <= 0.5 else 1 for x in y_pred]
    else:
        y_pred = t.svm_predictions(model, x, y_test)

    print(f"\nTS {model_name} REPORT:\n")
    print(classification_report(y_test, y_pred, digits=4))


def _is_svd(model_name: str) -> bool:
    return model_name[0] == "s" and model_name[2] == "d"


if __name__ == "__main__":
    main()
