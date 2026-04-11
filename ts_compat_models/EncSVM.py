import numpy as np

class EncSVM:
    def __init__(self, ts_svm):
        self.w = ts_svm.w.data.tolist()
        self.b = ts_svm.b.data.tolist()

    def enc_prelim_predict(self, enc_x_i):
        return enc_x_i.dot(self.w) - self.b

    def plaintext_predict(self, x_i):
        y = int(np.sign(np.dot(self.w, x_i) - self.b))
        if y == -1: y = 0

        return y