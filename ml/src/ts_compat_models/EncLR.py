import numpy as np
import src.client as client

"""
Because of how 'magical' the tutorial was for the Logistic regression implementation, we need
a new class to actually save the model (I don't actually know lol, but I am highly suspicious that the
LogisticReg.py won't be serialisable and save with pickle, since it sets up a whole torch module behind the
scenes).

This is used to be able to save the model (and load it later), with its weights and bias, and to be able
to perform encrypted inferences.
"""


class EncLR:
    def __init__(self, trained_lr):
        self.trained_lr = trained_lr  # so we can call testAccuracy
        self.w = trained_lr.log_reg.weight.data.tolist()[0]
        self.b = trained_lr.log_reg.bias.data.tolist()

    # exactly the same as svm. Client later applies sigmoid to finally classify.
    def enc_prelim_predict(self, enc_x_i):
        return enc_x_i.dot(self.w) - self.b

    # for plaintext
    def plaintext_predict(self, x_i):
        prelim_y = np.dot(x_i, self.w) - self.b
        # print(f"plain log prelim_y = {prelim_y}")
        y = client.ts_client_finish_prediction_log(prelim_y)
        # print(f"post sigmoid y = {y}")

        if y < 0.5:
            y = 0
        else:
            y = 1

        return y

    def test_accuracy(self, X_test):
        y_pred = self.trained_lr(X_test)

        return y_pred
