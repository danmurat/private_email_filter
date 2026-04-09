import tenseal as ts
from LinearSVM import LinearSVM
"""
Handles encrypted inference of our SVM model
NOT NEEDED. DELETE
"""
class EncSVM:
    def __init__(self, ts_svm: LinearSVM):
        self.ts_svm = ts_svm
        self.w = ts_svm.get_w()
        self.b = ts_svm.get_b()
        # potentially just pass w/b into obj, rather than the model object (not sure if we get any efficiency?)
        # i think the copying here however would take up a lot of space? w can be very large?
        # we could just do ts_svm.get_w() in our computing part?


