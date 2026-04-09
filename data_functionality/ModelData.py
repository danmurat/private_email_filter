"""
Holds (already pre-processed) training and testing data that we'll save and load as a pkl file.
"""

class ModelData:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def get_X_train(self):
        return self.X_train

    def get_y_train(self):
        return self.y_train

    def get_X_test(self):
        return self.X_test

    def get_y_test(self):
        return self.y_test