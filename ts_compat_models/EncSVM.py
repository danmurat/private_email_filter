import numpy as np
import torch

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


    # PASS REGULAR ARRAY HERE, NOT TORCH TENSOR
    def testAcc(self, X, y):
        correct_counter = 0
        length = len(X)
        y_pred_list = []


        for i in range(length):
            # sign just turns anything positive to 1 and negative to -1
            # item gets rid of the computational graph we create when using "grad"
            y_pred = int(np.sign(np.dot(self.w, X[i]) - self.b))
            if y_pred == -1:
                y_pred = 0
            y_pred_list.append(y_pred)
            if y_pred == y[i]:
                correct_counter += 1

        accuracy = correct_counter / length
        print(f"SVM Accuracy = {accuracy}")
        #print(y_pred_list)
        return y_pred_list
