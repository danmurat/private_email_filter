from TenSealModels import TenSealModels
import util
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# DATA
model_data = util.loadModelPickle(util.model_data_path())
reduced_model_data = util.loadModelPickle(util.reduced_model_data_path())

X_test = model_data.get_X_test()
y_test = model_data.get_y_test()
X_test = X_test.toarray() # remember these are sparse when loaded
red_X_test = reduced_model_data.get_red_X_test()

t_X_train, t_y_train, t_X_test, t_y_test, t_red_X_train, t_red_X_test = util.convertToTorchTensors(np.zeros((0,0)), np.zeros((0,0)), X_test, y_test, np.zeros((0,0)), red_X_test)

"""
Seperately evaluating predictive performance of all plaintext models.
(encrypted prediction is equivalent, so no need to waste time).
"""
def main():
    zama_report("svd_log")
    zama_report("svd_svm")
    ts_report("svd_log")
    ts_report("svd_svm")

def zama_report(model_name):
    model = util.loadModelPickle(f"zama_plain_models/{model_name}")
    y_pred = model.predict(red_X_test) if _is_svd(model_name) else model.predict(X_test)

    print(f"\nZAMA {model_name} REPORT:\n")
    print(classification_report(y_test, y_pred, digits=4))

def ts_report(model_name):
    t = TenSealModels() # to run the accuracy tests
    model = util.loadModelPickle(f"ts_plain_models/{model_name}")

    is_log = model_name[0] == 'l' or (len(model_name) > 3 and model_name[4] == 'l') # not sure if [5] doesn't exist, will break?

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

def _is_svd(model_name):
    return model_name[0] == 's' and model_name[2] == 'd'


if __name__ == "__main__":
    main()