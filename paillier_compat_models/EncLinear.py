import numpy as np

"""
Turns out all the work we did getting the tenseal models to work, is pretty much also usable for python-pallier!
We just get the model weights and multiply them by the encrypted email vector!

Just this single class can represent both logreg and svm, because all that's needed is (wx - b) !
"""
class EncLinear:
    def __init__(self, model):
        self.w = model.w
        self.b = model.b

    def enc_prelim_predict(self, enc_x_i):
        return np.dot(enc_x_i, self.w) - self.b[0] # paillier doesn't like enc_list - [b]. enc_list - b works